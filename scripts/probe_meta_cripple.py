#!/usr/bin/env python3
"""Is the `meta` (continual) strategy IMPLICITLY CRIPPLED by the trainer, or genuinely worse?

    # mechanism, from the campaign's own checkpoints -- no training, ~1 detector-call batch per design
    python scripts/probe_meta_cripple.py sensitivity =enzyme_inhib \
        --run output/campaign-inhib/126382657/meta --output output/screen/meta-sens-meta.json

    # the A/B/C/D/E table at one k, on a reconstructed pool
    python scripts/probe_meta_cripple.py table =enzyme_inhib \
        --results output/campaign-inhib/126382657/from_scratch/results.json --k 5 \
        --output output/screen/meta-cripple-k5.json

THE HYPOTHESIS. `combine_scaled` puts each experiment's four DESIGN coordinates in the last four
feature channels. `EnsembleSetBlock` inserts `nnx.Dropout` BEFORE EVERY shared linear layer including
the FIRST, so at `p_dropout: 0.1` the design channels are dropped 10% of the time -- INPUT dropout on
the design. For `from_scratch` the pool holds ONE design, so those channels are constant and dropping
them costs nothing (the constant goes into a bias, the weights go to zero). For `meta` the pool
ACCUMULATES designs, so those channels carry the signal the method depends on and the same dropout is
a shrinkage penalty on design sensitivity -- pulling the network toward the design-MARGINAL answer,
more so as designs accumulate. A competing, non-exclusive hypothesis: the network is simply
UNDER-CAPACITY for many designs.

WHAT THIS SCRIPT MEASURES, and nothing else. It is a TRAINER DIAGNOSTIC on RECONSTRUCTED pools. No
number here is a benchmark result about either candidate, and nothing here changes `p_dropout`, the
dropout placement, or any config.

MODE `sensitivity` -- the mechanism, head-on, with no training at all. Every per-design checkpoint a
campaign run wrote is restored and probed at its OWN design on a common held-out event draw:

  * `design_tv`   -- mean total-variation move of the predicted class distribution when every design
    channel is pushed 0.1 of its range INWARD (toward the centre of the scaled cube, so nothing
    clips). This is the network's design sensitivity in the units the prediction is read in.
  * `measure_tv`  -- the same for the MEASUREMENT channels, pushed 0.1 of THEIR range (they live on
    [-1, 1], so the step is 0.2). The CONTROL: it says whether a fall in `design_tv` is the design
    channels specifically or the whole output flattening.
  * `blind_delta` -- the loss penalty for replacing every design channel by 0.5 (the centre of the
    cube), i.e. how much the network's accuracy actually RELIES on knowing the design. A network
    pulled onto the design-marginal answer pays nothing to be blinded.

MODE `table` -- A/B/C/D/E at one iteration `k` of a recorded BO trajectory, every arm trained until
its validation loss stops improving (see :func:`settled` -- NOT the trainer's own plateau test, whose
0.008 tolerance is four times the difference this table must resolve) and scored on the SAME held-out
draw AT DESIGN k (what BO needs is an accurate score for the design under evaluation):

  A `from_scratch`  design k's own N samples          C vs A: the crippling itself
  B `from_scratch`  the same at 2N                    B vs A: how much is just data volume
  C `meta`          designs 1..k, trajectory counts   D vs C: under-training vs BIAS
  D `meta`          the same with every count doubled
  E XGBoost         C's pool, design columns INCLUDED E vs C: is the NETWORK the limitation

The pool is reconstructed EXACTLY: a recorded `spent` decomposes uniquely into the trainer's own
`n0 + m*n_increment` train rows plus their `round(., val_ratio)` validation rows, so arm C sees the
window sizes the run really used, in the order it really used them, with the continual trainer's own
replay sampler and one persistent network carried across them.
"""

import argparse
import copy
import gc
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("AGG")

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.detector
from detopt.nn import from_config as regressor_from_config
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils import io
from detopt.utils.pools import Pool
from detopt.utils.training import masked_mean_sem


# --------------------------------------------------------------------------------------------- #
# Config + trajectory
# --------------------------------------------------------------------------------------------- #
def load_config(name):
  """The RUN config and its detector config, resolved the way gearup resolves them."""
  import yaml
  name = name.lstrip('=')
  with open(f'config/{name}.yaml') as f:
    config = yaml.safe_load(f)
  detector_config = config['detector']
  if isinstance(detector_config, str):
    with open(f'config/detector/{detector_config}.yaml') as f:
      detector_config = yaml.safe_load(f)
  return config, detector_config


def window_counts(spent, n0, n_increment, val_fraction):
  """A recorded ``spent`` -> the ``(n_train, n_val)`` rows the trainer actually appended.

  The trainer adds ``n0`` train rows and ``round(n0 * ratio)`` validation rows, then ``m`` rounds of
  ``n_increment`` + ``round(n_increment * ratio)``, so ``spent`` determines ``m`` exactly. When it
  does not (a design cut short by the budget), fall back to the plain ``val_fraction`` split.
  """
  ratio = val_fraction / (1.0 - val_fraction)
  v0, v1 = round(n0 * ratio), round(n_increment * ratio)
  m, rest = divmod(int(spent) - (n0 + v0), n_increment + v1)
  if rest != 0 or m < 0:
    n_train = int(round(spent * (1.0 - val_fraction)))
    return n_train, int(spent) - n_train
  return int(n0 + m * n_increment), int(v0 + m * v1)


# --------------------------------------------------------------------------------------------- #
# Sampling. Identical events to the trainer's own fill (the detector is a pure function of
# (design, event_index)); only the CHUNK differs, and the detector is latency-bound, so a larger
# chunk is the same data several times faster.
# --------------------------------------------------------------------------------------------- #
def fill(detector, pool, design, n_to_add, index_array, chunk):
  added = 0
  while added < n_to_add:
    k = min(chunk, n_to_add - added)
    start = pool.current
    event_index = index_array[start:start + k]
    if event_index.shape[0] < k:
      raise RuntimeError(f'event index exhausted: need {k} from {start}, have {event_index.shape[0]}')
    design_b = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (k,) + jnp.asarray(a).shape), design)
    _gt, event, mask, target = detector(design_b, event_index)
    pool.append(event, mask, target, design_b)
    added += k


def holdout_index(n, budget_index, seed):
  """``n`` event indices DISJOINT from every index the pools will consume.

  The analytic source draws its indices uniformly from ``[0, 2**31)``, so a fresh draw overlaps the
  training draw by ~n*budget/2**31 events; the overlap is removed outright rather than argued to be
  small.
  """
  rng = np.random.default_rng(int(seed) + 991)
  used = np.unique(np.asarray(budget_index, np.int64))
  out = np.empty(0, np.int64)
  while out.shape[0] < n:
    draw = rng.integers(0, 2**31 - 1, size=2 * n, dtype=np.int64)
    out = np.unique(np.concatenate([out, np.setdiff1d(draw, used, assume_unique=False)]))
  return out[:n]


# --------------------------------------------------------------------------------------------- #
# Design sensitivity. The design channels of `combine_scaled` are the LAST `n_design_fields` of the
# feature axis (each experiment's own coordinates); the measurement channels are the rest.
# --------------------------------------------------------------------------------------------- #
def perturbed_features(features, first, last, delta, centre):
  """``features`` with channels ``[first, last)`` pushed ``delta`` INWARD, i.e. toward ``centre``.

  The design channels live on [0, 1] (centre 0.5) and the measurements on [-1, 1] (centre 0.0), and
  the sign is chosen per value so the step never leaves the range and never clips. BO piles designs
  onto the box CORNERS, where a fixed-sign step would clip on half of them and read as reduced
  sensitivity that is really a saturated perturbation.
  """
  block = features[..., first:last]
  step = jnp.where(block <= centre, delta, -delta)
  return features.at[..., first:last].set(block + step)


def input_weight_norms(reg, n_design):
  """RMS first-layer weight on the DESIGN input channels and on the MEASUREMENT ones.

  The quantity input dropout shrinks is precisely the weight on the dropped channel, so reading it
  straight off the parameters is the most direct form of the measurement -- no forward pass, no
  perturbation size to argue about. RMS PER WEIGHT (not a summed norm) so the two blocks are
  comparable despite having different channel counts. Returns ``(None, None)`` for an architecture
  whose first layer this does not describe.
  """
  from detopt.nn.set_regressor import EnsembleLinear
  try:
    block = reg.blocks[0]
    layers = [layer for layer in block.shared if isinstance(layer, EnsembleLinear)]
    first = layers[0] if len(layers) > 0 else block.output
    kernel = first.kernel[...]  # (in, out) or (members, in, out); the input axis is -2
  except (AttributeError, IndexError, TypeError):
    return None, None
  design = float(jnp.sqrt(jnp.mean(jnp.square(kernel[..., -n_design:, :]))))
  measure = float(jnp.sqrt(jnp.mean(jnp.square(kernel[..., :-n_design, :]))))
  return design, measure


def sensitivity(reg, detector, event, mask, target, design_scaled, *, delta_design, delta_measure, blind_value):
  """Design sensitivity, its measurement-channel control, and the cost of blinding the design."""
  members = reg.ensemble()
  n_design = len(detector.design_spec()._fields)
  features = detector.combine_scaled(event, jnp.asarray(design_scaled, jnp.float32), mask=mask)
  emask = detector.element_mask(event, mask)
  normalised = detector.normalize_target(target)
  n_features = features.shape[-1]

  def predict(f):
    if members is None:
      return reg(f, emask, deterministic=True)
    return reg(jnp.broadcast_to(f, (members,) + f.shape), jnp.broadcast_to(emask, (members,) + emask.shape),
               deterministic=True).mean(axis=0)

  logits = predict(features)
  probabilities = jax.nn.softmax(logits, axis=-1)

  def total_variation(f):
    moved = jax.nn.softmax(predict(f), axis=-1)
    return float(jnp.mean(0.5 * jnp.sum(jnp.abs(moved - probabilities), axis=-1)))

  design_tv = total_variation(perturbed_features(features, n_features - n_design, n_features, delta_design, 0.5))
  measure_tv = total_variation(perturbed_features(features, 0, n_features - n_design, delta_measure, 0.0))

  # BLINDING: the design channels replaced by a constant. How much the loss RISES is how much the
  # network's accuracy actually relies on knowing the design. `design_offset` is how far this design
  # sits from that constant (RMS over the scaled coordinates) -- blinding a design that already sits
  # at the reference moves nothing, so the offset is what `blind_delta` has to be read against.
  blinded = features.at[..., n_features - n_design:].set(blind_value)
  design_block = features[..., n_features - n_design:]
  loss_true = float(jnp.mean(detector.loss(logits, normalised)))
  loss_blind = float(jnp.mean(detector.loss(predict(blinded), normalised)))
  design_weight, measure_weight = input_weight_norms(reg, n_design)
  return {
    'design_tv': design_tv, 'measure_tv': measure_tv,
    'tv_ratio': design_tv / measure_tv if measure_tv > 0 else float('nan'),
    'design_offset': float(jnp.sqrt(jnp.mean(jnp.square(design_block - blind_value)))),
    'loss': loss_true, 'loss_blind': loss_blind, 'blind_delta': loss_blind - loss_true,
    'design_weight': design_weight, 'measure_weight': measure_weight,
    'weight_ratio': (design_weight / measure_weight) if (measure_weight is not None and measure_weight > 0)
    else float('nan'),
  }


# --------------------------------------------------------------------------------------------- #
# MODE `sensitivity`: every per-design checkpoint of one run.
# --------------------------------------------------------------------------------------------- #
def run_sensitivity(arguments):
  config, detector_config = load_config(arguments.config)
  detector = detopt.detector.from_config(detector_config)
  from detopt.utils.config import resolve_device
  device = resolve_device(arguments.device if arguments.device is not None else config.get('device'))

  with open(os.path.join(arguments.run, 'results.json')) as f:
    recorded = io.check_bo_results(json.load(f)['results'], arguments.run)
  checkpoints = os.path.join(arguments.run, 'checkpoints')

  index = np.random.default_rng(arguments.seed).integers(0, 2**31 - 1, size=arguments.n_events, dtype=np.int64)
  rows = []
  for entry in recorded:
    iteration = int(entry['iteration'])
    if iteration % arguments.every != 0 and iteration != len(recorded) - 1:
      continue
    path = os.path.join(checkpoints, f'design_{iteration:04d}')
    if not os.path.isdir(path):
      print(f'  [skip] iteration {iteration}: no checkpoint at {path}')
      continue
    manager = io.get_checkpointer(path)
    if manager.latest_step() is None:
      manager.close()
      continue
    pure_params, pure_state, ckpt_design, _aux = io.restore_training_checkpoint(manager)
    manager.close()
    design_scaled = np.asarray(entry['x_scaled'], np.float32)
    if not np.allclose(np.asarray(ckpt_design['scaled'], np.float32), design_scaled, atol=1e-5):
      raise ValueError(f'iteration {iteration}: checkpoint design differs from results.json')
    # A FRESH abstract module per checkpoint: `replace_by_pure_dict` mutates the state it is handed,
    # so reusing one across restores would leave a checkpoint's weights standing wherever the next
    # pure dict happened not to reach.
    graphdef, params, state = nnx.split(
      regressor_from_config(detector, config=config['regressor'], rngs=nnx.Rngs(0)), nnx.Param, nnx.Variable
    )
    nnx.replace_by_pure_dict(params, pure_params)
    nnx.replace_by_pure_dict(state, pure_state)
    restored = nnx.merge(graphdef, jax.device_put(params, device), jax.device_put(state, device))

    design = detector.to_nominal(design_scaled)
    design_b = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a)[None], (len(index),) + jnp.asarray(a).shape), design)
    _gt, event, mask, target = detector(design_b, index)
    row = sensitivity(
      restored, detector, event, mask, target, design_scaled,
      delta_design=arguments.delta, delta_measure=2.0 * arguments.delta, blind_value=0.5
    )
    row.update({'iteration': iteration, 'reported_loss': float(entry['loss']), 'spent': int(entry['spent'])})
    rows.append(row)
    print(f'  iter {iteration:>3d}  design_tv {row["design_tv"]:.4f}  measure_tv {row["measure_tv"]:.4f}  '
          f'tv_ratio {row["tv_ratio"]:.3f}  w_ratio {row["weight_ratio"]:.3f}  loss {row["loss"]:.4f}  '
          f'blind_delta {row["blind_delta"]:+.4f}  offset {row["design_offset"]:.3f}', flush=True)
    with open(arguments.output, 'w') as f:
      json.dump({'run': arguments.run, 'n_events': int(arguments.n_events), 'delta': float(arguments.delta),
                 'rows': rows}, f, indent=2, default=float)
  print(f'wrote {arguments.output}')


# --------------------------------------------------------------------------------------------- #
# MODE `table`: A/B/C/D/E at one k.
# --------------------------------------------------------------------------------------------- #
def with_overrides(run_config, cell):
  """A copy of the run config with this CELL's overrides applied. Command-line only -- the shipped
  config on disk is never touched.

  Recognised keys: ``p_dropout`` and ``features`` (the regressor's block definitions) on the
  regressor, ``weight_decay`` on the optimiser. ``data_scale`` is not a config value and is applied
  to the reconstructed segment counts instead.
  """
  config = copy.deepcopy(run_config)
  (regressor_name,), = (config['regressor'].keys(),)
  (optimizer_name,), = (config['training']['optimizer'].keys(),)
  if 'p_dropout' in cell:
    config['regressor'][regressor_name]['p_dropout'] = float(cell['p_dropout'])
  if 'features' in cell:
    config['regressor'][regressor_name]['features'] = [[int(w) for w in block] for block in cell['features']]
  if 'weight_decay' in cell:
    config['training']['optimizer'][optimizer_name]['weight_decay'] = float(cell['weight_decay'])
  return config


def count_parameters(detector, regressor_config, seed=0):
  """Total learnable parameters of a regressor config, ENSEMBLE MEMBERS INCLUDED.

  Counted, not derived: the two blocks have different shapes, every linear carries a bias, the
  learnable activation adds two gains per unit, and ``n_models`` multiplies all of it -- so a width
  factor does not map to a parameter factor by any rule worth trusting.
  """
  regressor = regressor_from_config(detector, config=regressor_config, rngs=nnx.Rngs(seed))
  _, params, _ = nnx.split(regressor, nnx.Param, nnx.Variable)
  return int(sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(params)))


def build_trainer(kind, detector, run_config, *, max_train, budget, steps_per_epoch, seed, device, replay_weight=None):
  """A trainer whose window cap fits the largest segment of this arm, with the EPOCH LENGTH pinned.

  ``iteration_limit`` normally does double duty -- the per-design window cap AND the epoch length
  (``steps_per_epoch = iteration_limit // batch``). An arm at 2x data would then also get 2x longer
  epochs, which is a second difference on top of the one being measured, so the epoch length is set
  to the campaign's own value for every arm and the kernels are rebuilt on it.
  """
  config = copy.deepcopy(run_config)
  training = config['training']
  n0, n_increment = int(training['n0']), int(training['n_increment'])
  limit = n0 + math.ceil(max(0, int(max_train) - n0) / n_increment) * n_increment
  training['iteration_limit'] = int(limit)
  training['budget'] = int(budget)
  if device is not None:
    config['device'] = device
  # `_DesignBase.from_config` spreads the `training` block onto the constructor, so the continual
  # trainer's keyword-only `replay_weight` travels there. It is NOT set for the per-design trainers:
  # they have no replay half, and `DesignTrainer.__init__` would reject the argument.
  if kind == 'meta' and replay_weight is not None:
    training['replay_weight'] = float(replay_weight)
  cls = ContinualTrainer if kind == 'meta' else DesignTrainer
  trainer = cls.from_config(detector, config, checkpoint_dir=None, seed=seed)
  if trainer.iteration_limit < max_train:
    raise RuntimeError(f'window cap {trainer.iteration_limit} below the largest segment {max_train}')
  trainer.steps_per_epoch = int(steps_per_epoch)
  trainer._build_kernels(seed)
  return trainer


def settled(history, block, tolerance):
  """True when the last ``block`` epochs did not improve on the ``block`` before them by ``tolerance``.

  test declares a plateau when the fitted drop over ``patience`` epochs falls under
  difference this table has to resolve; run against it, every arm stopped at epoch 17 -- the first
  epoch the test is allowed to fire -- and arm A came out 0.011 ABOVE the level the campaign itself
  reported for the same design, i.e. plainly still falling. Comparing two block MEANS instead of
  fitting a slope averages the per-epoch evaluation noise down by ``sqrt(block)``, so a tolerance well
  below the effect size is measurable rather than swamped.
  """
  if len(history) < 2 * block:
    return False
  recent = float(np.mean(history[-block:]))
  earlier = float(np.mean(history[-2 * block:-block]))
  return earlier - recent < tolerance


def train_segment(trainer, network, key, w0_train, n_train, w0_val, n_val, knobs):
  """Train one segment until the validation loss stops improving (see :func:`settled`).

  Every arm gets the SAME epoch length (``steps_per_epoch`` is pinned) and the same stopping rule, so
  the arms differ only in what is in the pool and whether the network was carried over. The
  best-validation network is kept alongside the final one: with the data fixed, a long run can start
  overfitting, and which of the two is scored should be visible rather than assumed.
  """
  params, state, opt_state = network
  tp, vp = trainer.train_pool, trainer.val_pool
  w0_t, w0_v = jnp.int32(w0_train), jnp.int32(w0_val)
  train_history, val_history = [], []
  train_sem = val_sem = float('nan')
  best = (float('inf'), 0, params, state)
  for epoch in range(knobs['max_epochs']):
    key, subkey = jax.random.split(key)
    params, state, opt_state, _ = trainer._train_epoch(
      params, state, opt_state, subkey, w0_t, jnp.int32(n_train), tp.buffers()
    )
    train_mean, train_sem = masked_mean_sem(trainer._eval_train(params, state, tp.buffers(), w0_t), n_train)
    val_mean, val_sem = masked_mean_sem(trainer._eval_val(params, state, vp.buffers(), w0_v), n_val)
    train_history.append(float(train_mean))
    val_history.append(float(val_mean))
    train_sem, val_sem = float(train_sem), float(val_sem)
    if float(val_mean) < best[0]:
      best = (float(val_mean), epoch + 1, params, state)
    if epoch + 1 < knobs['min_epochs']:
      continue
    if settled(val_history, knobs['block'], knobs['tolerance']):
      break
  block = knobs['block']
  drift = float(np.mean(val_history[-2 * block:-block]) - np.mean(val_history[-block:])) if len(
    val_history) >= 2 * block else float('nan')
  return (params, state, opt_state), best[2:], {
    'epochs': len(train_history), 'train': train_history[-1], 'val': val_history[-1],
    'train_sem': train_sem, 'val_sem': val_sem, 'n_train': int(n_train), 'n_val': int(n_val),
    # The DEMONSTRATION of convergence: how much the validation loss still moved over the last block
    # of epochs, in the same units as the table. It has to be small against the arm-to-arm gaps.
    'val_drift': drift, 'best_val': best[0], 'best_epoch': best[1],
  }


def run_arm(kind, label, detector, run_config, segments, holdout, *, steps_per_epoch, knobs, seed, device, chunk,
            replay_weight=None, shared=None):
  """Train one arm over its segments and score it on the shared held-out set at the LAST segment.

  ``shared`` is an already-filled ``{'train', 'val', 'offsets'}`` from an earlier arm with the SAME
  segments and the same seed. A hyper-parameter sweep changes the network and the optimiser but not
  one event: the pool depends only on (designs, counts, seed), all of which are fixed across a grid,
  so re-simulating it per cell would be tens of millions of identical detector calls. Passing it in
  reuses the rows and the window offsets verbatim; passing ``None`` fills fresh and the filled pools
  are returned for the next cell.
  """
  n_train_total = sum(s['n_train'] for s in segments)
  n_val_total = sum(s['n_val'] for s in segments)
  budget = int(math.ceil((n_train_total + n_val_total) * 1.02)) + 4096
  trainer = build_trainer(
    kind, detector, run_config, max_train=max(s['n_train'] for s in segments), budget=budget,
    steps_per_epoch=steps_per_epoch, seed=seed, device=device, replay_weight=replay_weight
  )
  if shared is not None:
    # Drop the freshly allocated (empty) pools and adopt the filled ones.
    trainer.train_pool, trainer.val_pool = shared['train'], shared['val']
  if trainer.val_iteration_limit < max(s['n_val'] for s in segments):
    raise RuntimeError('validation window cap below the largest validation segment')

  # WHAT THE LOSS ACTUALLY RECEIVES. `_sample_weights` is read back off the built trainer and its
  # two halves reported, rather than the requested number being assumed to have arrived: the weighted
  # path is new, and a knob that silently fails to reach the loss would produce a null that looks
  # exactly like the interesting answer.
  sample_weights = trainer._sample_weights()
  if sample_weights is None:
    weight_report = {'uniform': True}
  else:
    w = np.asarray(sample_weights, np.float64)
    n_cur = trainer.batch - trainer.batch // 2
    current, replay = w[:, :n_cur], w[:, n_cur:]
    weight_report = {
      'uniform': False, 'current': float(current.mean()), 'replay': float(replay.mean()),
      'ratio': float(current.mean() / replay.mean()), 'mean': float(w.mean()),
      'spread': float(current.std() + replay.std()),  # 0 = each half is a single constant, as intended
    }
    print(f'  [{label}] batch weights: current {weight_report["current"]:.4f} replay '
          f'{weight_report["replay"]:.4f} ratio {weight_report["ratio"]:.3f} mean {weight_report["mean"]:.6f} '
          f'spread {weight_report["spread"]:.2e}', flush=True)

  reg = regressor_from_config(detector, config=trainer.regressor_config, rngs=nnx.Rngs(seed))
  reg_def = nnx.split(reg, nnx.Param, nnx.Variable)[0]
  eval_holdout = trainer._build_eval(reg_def, holdout['n'])

  sequence = np.random.SeedSequence(seed)
  key = jax.random.PRNGKey(seed)
  network = None
  history = []
  offsets = []
  started = time.time()
  for i, segment in enumerate(segments):
    design = detector.to_nominal(segment['x_scaled'])
    t0 = time.time()
    if shared is None:
      w0_train, w0_val = trainer.train_pool.current, trainer.val_pool.current
      fill(detector, trainer.train_pool, design, segment['n_train'], trainer._train_index, chunk)
      fill(detector, trainer.val_pool, design, segment['n_val'], trainer._val_index, chunk)
    else:
      w0_train, w0_val = shared['offsets'][i]
    offsets.append((w0_train, w0_val))
    sampled = time.time() - t0
    # PAIRED INITIALISATION. The continual trainer builds its persistent net from `seed` itself; a
    # fresh per-design net would otherwise start from a spawned sub-seed, so A and C would differ by
    # their draw as well as by their strategy. Handing the same initial params to the from_scratch
    # arm makes initialisation drop out of the C - A difference within a seed.
    initial = trainer._build_regressor(seed)[1] if kind != 'meta' else None
    network = trainer._init_design_network(sequence.spawn(1)[0], initial)
    key, subkey = jax.random.split(key)
    network, best, record = train_segment(
      trainer, network, subkey, w0_train, segment['n_train'], w0_val, segment['n_val'], knobs
    )
    trainer._persist_network(*network)
    record.update({'segment': i, 'sample_s': sampled, 'total_s': time.time() - t0})
    history.append(record)
    print(f'  [{label}] segment {i + 1}/{len(segments)}  n_train={segment["n_train"]:>7d}  '
          f'epochs={record["epochs"]:>3d}  train={record["train"]:.4f} val={record["val"]:.4f}  '
          f'drift={record["val_drift"]:+.5f}  ({record["total_s"]:.0f}s, sample {sampled:.0f}s)', flush=True)

  params, state, _ = network
  losses = eval_holdout(params, state, holdout['pool'].buffers(), jnp.int32(0))
  mean, sem = masked_mean_sem(losses, holdout['n'])
  best_losses = eval_holdout(best[0], best[1], holdout['pool'].buffers(), jnp.int32(0))
  best_mean, _ = masked_mean_sem(best_losses, holdout['n'])
  last = history[-1]
  sensitivities = sensitivity(
    nnx.merge(reg_def, params, state), detector, holdout['event'], holdout['mask'], holdout['target'],
    segments[-1]['x_scaled'], delta_design=knobs['delta'], delta_measure=2.0 * knobs['delta'], blind_value=0.5
  )
  row = {
    'arm': label, 'kind': kind, 'n_designs': len(segments), 'n_train_total': n_train_total,
    'n_val_total': n_val_total, 'calls': n_train_total + n_val_total,
    'replay_weight': replay_weight, 'batch_weights': weight_report,
    'holdout_loss': float(mean), 'holdout_sem': float(sem), 'holdout_loss_best_val': float(best_mean),
    'train_loss': last['train'], 'val_loss': last['val'],
    'gap_train_val': last['val'] - last['train'], 'gap_train_holdout': float(mean) - last['train'],
    'epochs_last': last['epochs'], 'val_drift_last': last['val_drift'],
    'wall_s': time.time() - started, 'history': history,
  }
  row.update({f'sens_{k}': v for k, v in sensitivities.items()})
  print(f'  [{label}] HELD-OUT {mean:.4f} +- {sem:.4f} (best-val net {best_mean:.4f})  '
        f'train {last["train"]:.4f} val {last["val"]:.4f} drift {last["val_drift"]:+.5f}  '
        f'design_tv {sensitivities["design_tv"]:.4f}  blind_delta {sensitivities["blind_delta"]:+.4f}',
        flush=True)
  filled = {'train': trainer.train_pool, 'val': trainer.val_pool, 'offsets': offsets}
  return row, losses, trainer, filled


def pool_arrays(detector, pool, keep, chunk=131072):
  """The pool's rows ``keep`` as ``(flat features, class labels)`` on the host.

  The features are what the NETWORK sees -- ``combine`` of each event with ITS OWN stored design, so
  the four scaled design coordinates per experiment are columns of the table, flattened. That is the
  one thing ``detopt.bo.gbdt.sample_design`` deliberately drops (correct for a per-design estimator,
  wrong here: the question is whether an architecture-independent learner can use a MULTI-design
  pool at all).
  """
  event_buf, mask_buf, target_buf, design_buf = pool.buffers()
  features, labels = [], []
  for start in range(0, keep.shape[0], chunk):
    index = jnp.asarray(keep[start:start + chunk])
    event = jax.tree.map(lambda a: a[index], event_buf)
    mask = mask_buf[index]
    design = jax.tree.map(lambda a: a[index], design_buf)
    target = jax.tree.map(lambda a: a[index], target_buf)
    combined = detector.combine(event, design, mask=mask)
    features.append(np.asarray(combined, np.float32).reshape(index.shape[0], -1))
    labels.append(np.asarray(jnp.argmax(target.mechanism, axis=-1), np.int32))
  return np.concatenate(features), np.concatenate(labels)


def run_gbdt(detector, data, holdout, *, max_learners, seed):
  """Arm E: XGBoost on the SAME pool as arm C (same rows, same designs), design columns included."""
  import xgboost as xgb
  from detopt.bo.gbdt import _parameters

  train_features, train_labels, val_features, val_labels, share, n_designs = data
  n_classes = int(detector.n_classes)
  parameters = _parameters(learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40, n_threads=None, seed=seed)
  parameters.update({'objective': 'multi:softprob', 'num_class': n_classes, 'eval_metric': 'mlogloss'})
  train_matrix = xgb.DMatrix(train_features, label=train_labels)
  val_matrix = xgb.DMatrix(val_features, label=val_labels)
  started = time.time()
  model = xgb.train(parameters, train_matrix, num_boost_round=int(max_learners),
                    evals=[(val_matrix, 'val')], early_stopping_rounds=50, verbose_eval=False)
  stage = (0, int(model.best_iteration) + 1)
  scale = math.log(n_classes)

  def cross_entropy(matrix, labels):
    probabilities = model.predict(matrix, iteration_range=stage)
    picked = probabilities[np.arange(len(labels)), labels.astype(np.int64)]
    return -np.log(np.clip(picked, 1e-12, None)) / scale

  holdout_features = np.asarray(holdout['features'], np.float32).reshape(holdout['n'], -1)
  holdout_matrix = xgb.DMatrix(holdout_features, label=holdout['labels'])
  train_loss = cross_entropy(train_matrix, train_labels)
  val_loss = cross_entropy(val_matrix, val_labels)
  holdout_loss = cross_entropy(holdout_matrix, holdout['labels'])
  row = {
    'arm': 'E xgboost', 'kind': 'gbdt', 'n_designs': int(n_designs),
    'n_train_total': int(train_labels.shape[0]), 'n_val_total': int(val_labels.shape[0]),
    'calls': int(train_labels.shape[0] + val_labels.shape[0]),
    'holdout_loss': float(holdout_loss.mean()),
    'holdout_sem': float(holdout_loss.std() / math.sqrt(holdout_loss.size - 1)),
    'train_loss': float(train_loss.mean()), 'val_loss': float(val_loss.mean()),
    'gap_train_val': float(val_loss.mean() - train_loss.mean()),
    'gap_train_holdout': float(holdout_loss.mean() - train_loss.mean()),
    'n_learners': int(model.best_iteration) + 1, 'row_share': float(share), 'wall_s': time.time() - started,
  }
  print(f'  [E xgboost] HELD-OUT {row["holdout_loss"]:.4f} +- {row["holdout_sem"]:.4f}  '
        f'train {row["train_loss"]:.4f}  val {row["val_loss"]:.4f}  learners {row["n_learners"]}', flush=True)
  return row, holdout_loss


def _prepare(arguments):
  """Everything the `table` and `sweep` modes share: detector, reconstructed segments, stopping knobs,
  and the held-out set at design k. Returns ``(detector, config, segments, knobs, steps_per_epoch,
  holdout, device)``."""
  config, detector_config = load_config(arguments.config)
  detector = detopt.detector.from_config(detector_config)
  training = config['training']
  n0, n_increment = int(training['n0']), int(training['n_increment'])
  val_fraction = float(training.get('val_fraction', 0.25))
  steps_per_epoch = max(1, int(training['iteration_limit']) // int(training['batch']))
  knobs = {
    'min_epochs': int(arguments.min_epochs), 'max_epochs': int(arguments.max_epochs),
    'block': int(arguments.block), 'tolerance': float(arguments.tolerance), 'delta': float(arguments.delta),
  }

  with open(arguments.results) as f:
    recorded = io.check_bo_results(json.load(f)['results'], arguments.results)
  k = int(arguments.k)
  if k > len(recorded):
    raise ValueError(f'{arguments.results} has {len(recorded)} iterations, k={k} requested')
  segments = []
  for entry in recorded[:k]:
    n_train, n_val = window_counts(entry['spent'], n0, n_increment, val_fraction)
    if arguments.cap_train is not None and n_train > arguments.cap_train:
      n_val = max(1, int(round(n_val * arguments.cap_train / n_train)))
      n_train = int(arguments.cap_train)
    segments.append({'x_scaled': np.asarray(entry['x_scaled'], np.float32), 'n_train': n_train, 'n_val': n_val,
                     'spent': int(entry['spent']), 'reported_loss': float(entry['loss'])})
  design_k = segments[-1]
  print(f'k={k}: {sum(s["spent"] for s in segments)} recorded calls over {k} designs; '
        f'design k spent {design_k["spent"]} (reported loss {design_k["reported_loss"]:.4f})')

  from detopt.utils.config import resolve_device
  device = resolve_device(arguments.device if arguments.device is not None else config.get('device'))

  # The held-out set at design k: shared by every arm, and disjoint from every index any arm consumes.
  # Every arm's trainer draws its budget index from `default_rng(seed).integers(0, 2**31 - 1, size=.)`,
  # whose first n values do not depend on n -- so one long prefix covers every arm's train AND val
  # slice (they sit at different offsets of the same stream), with room to spare.
  total = 4 * sum(s['spent'] for s in segments) + 65536
  budget_index = np.random.default_rng(arguments.seed).integers(0, 2**31 - 1, size=total, dtype=np.int64)
  index = holdout_index(arguments.n_holdout, budget_index, arguments.seed)
  specs = (detector.event_spec(), jax.ShapeDtypeStruct((detector.combined_event_shape()[0],), jnp.int32),
           detector.target_spec(), detector.design_spec())
  holdout_pool = Pool(arguments.n_holdout, specs, device)
  fill(detector, holdout_pool, detector.to_nominal(design_k['x_scaled']), arguments.n_holdout, index, arguments.chunk)
  event, mask, target, _design = holdout_pool.buffers()
  holdout_features, holdout_labels = pool_arrays(detector, holdout_pool, np.arange(arguments.n_holdout))
  holdout = {
    'pool': holdout_pool, 'n': int(arguments.n_holdout), 'event': event, 'mask': mask, 'target': target,
    'features': holdout_features, 'labels': holdout_labels,
  }
  print(f'held-out: {arguments.n_holdout} events at design k, disjoint from all training indices')
  return detector, config, segments, knobs, steps_per_epoch, holdout, device


def run_table(arguments):
  detector, config, segments, knobs, steps_per_epoch, holdout, device = _prepare(arguments)
  k = int(arguments.k)
  design_k = segments[-1]
  doubled = [dict(s, n_train=2 * s['n_train'], n_val=2 * s['n_val']) for s in segments]

  wanted = set(arguments.arms)
  # One meta arm PER REPLAY WEIGHT. The first weight (1.0 by default) is arm C -- the control, whose
  # job is to reproduce the unweighted numbers through the new weighted code path; the rest are C'
  # variants that differ from it in nothing but what a replay row is worth in the gradient.
  weights = [float(w) for w in arguments.replay_weight]
  plan = [
    ('A', 'from_scratch', 'A from_scratch N', [design_k], None),
    ('B', 'from_scratch', 'B from_scratch 2N', [dict(design_k, n_train=2 * design_k['n_train'],
                                                     n_val=2 * design_k['n_val'])], None),
  ]
  for index, weight in enumerate(weights):
    tag = 'C' if index == 0 else f"C{'*' * index}"
    plan.append((tag, 'meta', f'{tag} meta 1..k w={weight:g}', segments, weight))
  plan.append(('D', 'meta', 'D meta 1..k 2x', doubled, weights[0]))

  rows, per_sample, gbdt_data = [], {}, None
  for tag, kind, label, arm_segments, weight in plan:
    if tag.rstrip('*') not in wanted:
      continue
    print(f'--- arm {label}: {sum(s["n_train"] + s["n_val"] for s in arm_segments)} detector calls', flush=True)
    row, losses, trainer, _filled = run_arm(
      kind, label, detector, config, arm_segments, holdout, steps_per_epoch=steps_per_epoch, knobs=knobs,
      seed=arguments.seed, device=arguments.device, chunk=arguments.chunk, replay_weight=weight
    )
    rows.append(row)
    per_sample[tag] = np.asarray(losses, np.float64)
    if tag == 'C' and 'E' in wanted:
      # Arm E is scored on ARM C'S OWN ROWS, pulled off the pool before it is freed -- literally the
      # same events, designs and train/val split, so E vs C isolates the estimator.
      rng = np.random.default_rng(arguments.seed + 17)
      n_train_total, n_val_total = row['n_train_total'], row['n_val_total']
      share = min(1.0, arguments.gbdt_max_rows / max(1, n_train_total))
      keep_train = np.sort(rng.choice(n_train_total, int(round(share * n_train_total)), replace=False))
      keep_val = np.sort(rng.choice(n_val_total, int(round(share * n_val_total)), replace=False))
      train_features, train_labels = pool_arrays(detector, trainer.train_pool, keep_train)
      val_features, val_labels = pool_arrays(detector, trainer.val_pool, keep_val)
      gbdt_data = (train_features, train_labels, val_features, val_labels, share, len(arm_segments))
      print(f'  [E] kept {train_labels.shape[0]} train / {val_labels.shape[0]} val rows of arm C '
            f'({100 * share:.0f}% of the pool)', flush=True)
    del trainer, losses
    gc.collect()
    _write(arguments, k, segments, rows, per_sample)
  if gbdt_data is not None:
    print('--- arm E xgboost (arm C\'s own rows, design columns included)', flush=True)
    row, losses = run_gbdt(detector, gbdt_data, holdout, max_learners=arguments.gbdt_max_learners,
                           seed=arguments.seed)
    rows.append(row)
    per_sample['E'] = np.asarray(losses, np.float64)
    _write(arguments, k, segments, rows, per_sample)
  print(f'wrote {arguments.output}')


def cell_name(cell):
  """A stable label for a cell, from whichever overrides it carries."""
  parts = []
  if 'p_dropout' in cell:
    parts.append(f"p={float(cell['p_dropout']):g}")
  if 'weight_decay' in cell:
    parts.append(f"wd={float(cell['weight_decay']):g}")
  if 'features' in cell:
    parts.append('feat=' + '_'.join('-'.join(str(int(w)) for w in block) for block in cell['features']))
  parts.append(f"data={float(cell.get('data_scale', 1.0)):g}x")
  return ' '.join(parts)


def run_sweep(arguments):
  """MODE `sweep`: arms A and C over a list of CELLS at one k -- both arms on every cell.

  A cell is a JSON dict of overrides: ``p_dropout`` / ``weight_decay`` (the regularisation question --
  the campaign shares one setting between two different fitting problems, and that setting was chosen
  for the design-dedicated one), ``features`` (the capacity question -- block widths, hence parameter
  count), and ``data_scale`` (a multiplier on every reconstructed per-design count, because a larger
  network on data sized for a smaller one is under-trained, not under-capacity, and a null would not
  distinguish the two).

  BOTH arms move together over the grid. Tuning only `meta` would move the handicap into the
  experiment design rather than measure it. Each arm's BEST cell is chosen on its own VALIDATION
  loss, never on the held-out score, so nothing is selected on the quantity being reported.
  """
  cells = [json.loads(text) for text in arguments.cell]
  if arguments.count_only:
    # PARAMETER COUNTS ONLY -- no detector calls, no training. Doubling widths roughly QUADRUPLES a
    # dense layer, so the width set that lands on a target parameter ratio has to be read off a count
    # rather than assumed from a scaling rule.
    config, detector_config = load_config(arguments.config)
    detector = detopt.detector.from_config(detector_config)
    base = count_parameters(detector, config['regressor'])
    print(f'baseline {config["regressor"]} -> {base} parameters')
    for cell in cells:
      n = count_parameters(detector, with_overrides(config, cell)['regressor'])
      print(f'  {cell_name(cell):<40s} {n:>8d} parameters  ratio {n / base:.4f}')
    return
  detector, config, segments, knobs, steps_per_epoch, holdout, device = _prepare(arguments)
  k = int(arguments.k)
  base_parameters = count_parameters(detector, config['regressor'])
  print(f'sweep at k={k}, seed {arguments.seed}: {len(cells)} cells x 2 arms; '
        f'baseline regressor has {base_parameters} parameters (all ensemble members)')

  rows, per_sample, shared = [], {}, {}
  for cell in cells:
    scale = float(cell.get('data_scale', 1.0))
    scaled = [dict(s, n_train=int(round(s['n_train'] * scale)), n_val=int(round(s['n_val'] * scale)))
              for s in segments]
    cell_config = with_overrides(config, cell)
    n_parameters = count_parameters(detector, cell_config['regressor'])
    name = cell_name(cell)
    print(f'=== cell {name}: {n_parameters} parameters ({n_parameters / base_parameters:.3f}x baseline), '
          f'data {scale:g}x', flush=True)
    for tag, kind, arm_segments in (('A', 'from_scratch', [scaled[-1]]), ('C', 'meta', scaled)):
      label = f'{tag} {name}'
      print(f'--- {label}', flush=True)
      row, losses, trainer, filled = run_arm(
        kind, label, detector, cell_config, arm_segments, holdout, steps_per_epoch=steps_per_epoch,
        knobs=knobs, seed=arguments.seed, device=arguments.device, chunk=arguments.chunk,
        replay_weight=1.0 if kind == 'meta' else None, shared=shared.get((tag, scale))
      )
      # One filled pool per (arm, data scale): a cell that changes only the network reuses the events.
      shared.setdefault((tag, scale), filled)
      row.update({
        'tag': tag, 'cell': name, 'overrides': cell, 'data_scale': scale, 'n_parameters': n_parameters,
        'parameter_ratio': n_parameters / base_parameters,
        'calls_per_design': row['calls'] / max(1, row['n_designs']),
      })
      rows.append(row)
      per_sample[f'{tag}|{name}'] = np.asarray(losses, np.float64)
      del trainer, losses
      gc.collect()
      _write_sweep(arguments, k, rows, per_sample, base_parameters)
  print(f'wrote {arguments.output}')


def _write_sweep(arguments, k, rows, per_sample, base_parameters):
  """Dump the grid, the paired C-A difference IN EACH CELL, and the best-vs-best comparison.

  Every difference is also given in units of ``loss_precision``: an event-level standard error over
  65536 held-out events answers "how precisely was this network's loss measured", which was never the
  question. What matters is the size against the tolerance the pipeline is run at, and against the
  network's own train/validation gap.
  """
  precision = float(arguments.loss_precision)
  paired = {}
  for key, losses in per_sample.items():
    tag, name = key.split('|', 1)
    if tag != 'C':
      continue
    baseline = per_sample.get(f'A|{name}')
    if baseline is None:
      continue
    difference = losses - baseline
    paired[name] = {'mean': float(difference.mean()), 'in_loss_precision': float(difference.mean() / precision),
                    'event_sem': float(difference.std() / math.sqrt(difference.size - 1))}

  best = {}
  for tag in ('A', 'C'):
    cells = [r for r in rows if r['tag'] == tag]
    if len(cells) == 0:
      continue
    pick = min(cells, key=lambda r: r['val_loss'])  # selected on VALIDATION, never on held-out
    best[tag] = {'cell': pick['cell'], 'val_loss': pick['val_loss'], 'holdout_loss': pick['holdout_loss'],
                 'gap_train_val': pick['gap_train_val'], 'n_parameters': pick['n_parameters'],
                 'calls_per_design': pick['calls_per_design']}
  if 'A' in best and 'C' in best:
    difference = per_sample[f"C|{best['C']['cell']}"] - per_sample[f"A|{best['A']['cell']}"]
    best['C_minus_A_at_own_best'] = {'mean': float(difference.mean()),
                                     'in_loss_precision': float(difference.mean() / precision)}
  with open(arguments.output, 'w') as f:
    json.dump({'results': arguments.results, 'k': k, 'n_holdout': int(arguments.n_holdout),
               'seed': int(arguments.seed), 'loss_precision': precision,
               'base_parameters': int(base_parameters), 'rows': rows,
               'paired_C_minus_A': paired, 'best': best}, f, indent=2, default=float)


def _write(arguments, k, segments, rows, per_sample):
  """Dump the table, plus the PAIRED difference of every arm against A on the shared held-out set."""
  paired = {}
  if 'A' in per_sample:
    for tag, losses in per_sample.items():
      if tag == 'A':
        continue
      difference = losses - per_sample['A']
      paired[tag] = {'mean': float(difference.mean()),
                     'sem': float(difference.std() / math.sqrt(difference.size - 1))}
  with open(arguments.output, 'w') as f:
    json.dump({
      'results': arguments.results, 'k': k, 'n_holdout': int(arguments.n_holdout), 'seed': int(arguments.seed),
      'segments': [{'n_train': s['n_train'], 'n_val': s['n_val'], 'spent': s['spent'],
                    'reported_loss': s['reported_loss']} for s in segments],
      'rows': rows, 'paired_vs_A': paired,
    }, f, indent=2, default=float)


# --------------------------------------------------------------------------------------------- #
def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument('mode', choices=('sensitivity', 'table', 'sweep'))
  parser.add_argument('config', help='gearup root token of the RUN config, e.g. =enzyme_inhib')
  parser.add_argument('--run', help='sensitivity: a finished run directory (results.json + checkpoints/)')
  parser.add_argument('--results', help='table: a finished results.json to take designs and counts from')
  parser.add_argument('--k', type=int, default=5, help='table: which iteration to reconstruct up to (1-based)')
  parser.add_argument('--arms', nargs='+', default=['A', 'B', 'C', 'D', 'E'])
  parser.add_argument('--replay-weight', type=float, nargs='+', default=[1.0],
                      help='what a REPLAY row is worth against a current-design row in the continual '
                           'trainer\'s gradient. One meta arm is run per value; the FIRST is arm C (1.0 '
                           'reproduces the unweighted behaviour exactly), the rest are C* variants.')
  parser.add_argument('--n-holdout', type=int, default=65536)
  parser.add_argument('--n-events', type=int, default=8192, help='sensitivity: events per checkpoint')
  parser.add_argument('--every', type=int, default=1, help='sensitivity: probe every Nth checkpoint')
  parser.add_argument('--delta', type=float, default=0.1,
                      help='perturbation, as a fraction of a design coordinate\'s full range')
  parser.add_argument('--cell', action='append', default=None,
                      help='sweep: one JSON dict of overrides per cell, repeatable. Keys: p_dropout, '
                           'weight_decay, features (block widths -> parameter count), data_scale (a '
                           'multiplier on every reconstructed per-design count). BOTH arms run every '
                           'cell. e.g. \'{"p_dropout":0.2,"weight_decay":1e-3}\' or '
                           '\'{"features":[[34,24],[24,34]],"data_scale":2.0}\'')
  parser.add_argument('--loss-precision', type=float, default=8.0e-3,
                      help='the pipeline tolerance every effect is reported as a multiple of')
  parser.add_argument('--count-only', action='store_true',
                      help='sweep: print each cell\'s parameter count and ratio, then exit')
  parser.add_argument('--min-epochs', type=int, default=64)
  parser.add_argument('--max-epochs', type=int, default=320)
  parser.add_argument('--block', type=int, default=24,
                      help='epochs per block in the stopping rule: training ends when the mean validation '
                           'loss over the last block did not beat the block before it by --tolerance')
  parser.add_argument('--tolerance', type=float, default=5.0e-4,
                      help='the stopping rule\'s improvement threshold. It must be WELL BELOW the arm-to-arm '
                           'difference the table has to resolve (~2e-3), which the trainer\'s own plateau '
                           'test (0.008 per window) is not.')
  parser.add_argument('--cap-train', type=int, default=None,
                      help='PLUMBING TEST ONLY: cap every segment at this many train rows. The reconstruction '
                           'is then no longer the trajectory\'s and no number it produces means anything.')
  parser.add_argument('--chunk', type=int, default=4096, help='detector call chunk (latency-bound: bigger is faster)')
  parser.add_argument('--gbdt-max-rows', type=int, default=400000)
  parser.add_argument('--gbdt-max-learners', type=int, default=300)
  parser.add_argument('--device', default=None)
  parser.add_argument('--seed', type=int, default=7)
  parser.add_argument('--output', required=True)
  arguments = parser.parse_args()
  os.makedirs(os.path.dirname(os.path.abspath(arguments.output)), exist_ok=True)

  if arguments.mode == 'sensitivity':
    if arguments.run is None:
      parser.error('--run is required for mode `sensitivity`')
    run_sensitivity(arguments)
    return
  if arguments.results is None:
    parser.error(f'--results is required for mode `{arguments.mode}`')
  if arguments.mode == 'sweep':
    if arguments.cell is None or len(arguments.cell) == 0:
      parser.error('mode `sweep` needs at least one --cell')
    run_sweep(arguments)
  else:
    run_table(arguments)


if __name__ == '__main__':
  main()
