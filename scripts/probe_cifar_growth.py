"""CIFAR-10 probe: the project's data-GROWTH procedure against ordinary FULL-DATA training.

ONE QUESTION, asked outside the project's own detectors: does the growth procedure land where
full-data training lands? Two runs of the SAME network from the SAME initial parameters on the same
split, with both exit tests imported from :mod:`detopt.utils.training` so the statistics are
identical machinery and not a re-implementation.

* BASELINE -- all 40000 training images from epoch 1, no growth. Its only exit is the training-loss
  plateau of :mod:`detopt.nn.trainer.fixed_window`:
  ``P(train change over +patience < loss_precision / 2) > 0.9`` on the Bayesian trend of the
  post-warmup training history with the measured per-epoch SEMs as known observation noise.
* GROWTH -- the window starts at ``n0`` images and grows by ``n_increment``, driven by the ordered
  checks of :mod:`detopt.nn.trainer.design`, mirrored clause for clause:
    (1) ``P(gap at +patience > loss_precision) > 0.9``                     -> add data
    (2) ``P(train change over +patience < loss_precision / 2) > 0.9``
        (2.1) ``|val - train| + err > loss_precision``                     -> add data
        (2.2) otherwise                                                    -> converged, return
    (3) otherwise                                                          -> keep training
  with ``gap = |val - train| + err``, ``err = hypot(train_sem, val_sem)``, and the history being the
  CURRENT ROUND's post-warmup epochs (a data addition restarts the round).
  ONE DEVIATION, and it is the only one: ``design.py`` RAISES when a data request finds the pool
  exhausted. Here that request instead switches the run permanently to the baseline's plateau test,
  evaluated on the final round's post-warmup history, and the run ends exactly as the baseline does.

THE REWIND (``--param-mix``, lambda), mirrored from ``design.py``'s data-addition branch. At every
SUCCESSFUL data addition the growth arm pulls the network back toward the one THIS RUN STARTED FROM,
``params = initial + (1 - lambda) * (current - initial)``, and REBUILDS the optimiser state, because
Adam's moments summarise a trajectory the rewind has just partly discarded. ``initial`` is this run's
own initial parameters and never a fresh draw, so the move discards a fraction of what was learned
without also landing in a different random basin. lambda = 0 is the CODE default and a pure carry;
this project's own campaigns run 0.1 (SHiP) to 0.25 (``config/enzyme_extremes.yaml``), so the carry
is the one condition the campaigns never use. Nothing else about the arm changes with lambda, and the
rewind never fires on the exhausted-pool path, where no addition happens.

THE EPOCH IS A FIXED NUMBER OF OPTIMISER STEPS IN BOTH RUNS -- ``n_train // batch`` -- as the
project's ``steps_per_epoch = iteration_limit // batch`` is. That is what makes two epoch-indexed
exit tests comparable at all; the growth run consequently makes many passes over a small window
early rather than taking fewer steps.

SPLIT. CIFAR-10's standard 50000/10000 split is kept. The 10000 test images are never trained or
validated on and are read exactly once per run, after it has stopped. The 50000 training images are
split 4:1 into 40000 train / 10000 validation by a permutation from ``--split-seed``, giving
40000 / 10000 / 10000 train / validation / test. Images are
uint8 on device and scaled to float32 in [0, 1] inside the network, so the normalisation is one code
path shared by both runs.

BOTH RUNS START FROM ONE DRAW of the initial parameters (``--init-seed``), so no part of the
comparison is an initialisation difference.

MODEL. ``conv3x3 -> conv3x3 -> maxpool2x2`` repeated until 1x1 (32 -> 16 -> 8 -> 4 -> 2 -> 1) at
widths 32, 64, 128, 128, 128 with ReLU, then a dense layer to 10 logits under softmax cross-entropy.
Trained functionally: parameters split out of the nnx module and merged inside ``jax.jit``.

REACHING ``--max-epochs`` IS NOT A RESULT. It is recorded as ``converged: false`` with its curve.

OUTPUTS under ``output/cifar-growth/``: ``baseline/`` and ``growth/`` each get ``result.json`` and
the project's own ``plot_iteration`` figure; ``overlay.png`` compares the two against epoch and
against images consumed.

NO NETWORK ACCESS: the data is read from ``data/cifar10/cifar10.npz`` on disk.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from detopt.utils.training import (bayesian_trend, masked_mean_sem, probability_above, probability_change_below)
from detopt.utils.viz.bo import plot_iteration

CHANNELS = (32, 64, 128, 128, 128)
N_CLASSES = 10
ARM_COLORS = {'baseline': 'tab:blue', 'growth': 'tab:red', 'growth-pmix025': 'tab:purple'}
FALLBACK_COLORS = ('tab:green', 'tab:orange', 'tab:brown', 'tab:olive')


class Stage(nnx.Module):
  """``conv3x3 -> relu -> conv3x3 -> relu -> maxpool2x2``; both convolutions at one width."""

  def __init__(self, in_channels: int, channels: int, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(int(in_channels), int(channels), kernel_size=(3, 3), padding='SAME', rngs=rngs)
    self.conv2 = nnx.Conv(int(channels), int(channels), kernel_size=(3, 3), padding='SAME', rngs=rngs)

  def __call__(self, h):
    h = jax.nn.relu(self.conv1(h))
    h = jax.nn.relu(self.conv2(h))
    return nnx.max_pool(h, window_shape=(2, 2), strides=(2, 2))


class ConvNet(nnx.Module):
  """Five stages from 32x32 down to 1x1, then a dense layer to ``N_CLASSES`` logits.

  Takes uint8 images ``(B, 32, 32, 3)`` and scales them to float32 in [0, 1] itself, so both runs
  normalise identically by construction. Emits LOGITS; the loss applies the softmax.
  """

  def __init__(self, channels=CHANNELS, n_classes: int = N_CLASSES, *, rngs: nnx.Rngs):
    widths = (3,) + tuple(int(c) for c in channels)
    self.stages = nnx.List([Stage(widths[i], widths[i + 1], rngs=rngs) for i in range(len(channels))])
    self.head = nnx.Linear(int(channels[-1]), int(n_classes), rngs=rngs)

  def __call__(self, images):
    h = images.astype(jnp.float32) / 255.0
    for stage in self.stages:
      h = stage(h)
    return self.head(h.reshape(h.shape[0], -1))


def load_split(path, split_seed, n_validation):
  """The fixed CIFAR-10 test set untouched; the 50000 training images permuted by ``split_seed`` and
  cut into train / validation."""
  with np.load(path) as data:
    train_images = np.asarray(data['train_images'], dtype=np.uint8)
    train_labels = np.asarray(data['train_labels'], dtype=np.int32)
    test_images = np.asarray(data['test_images'], dtype=np.uint8)
    test_labels = np.asarray(data['test_labels'], dtype=np.int32)
  order = np.random.default_rng(split_seed).permutation(train_images.shape[0])
  validation_index = order[:n_validation]
  train_index = order[n_validation:]
  return {
    'train_images': jnp.asarray(train_images[train_index]),
    'train_labels': jnp.asarray(train_labels[train_index]),
    'val_images': jnp.asarray(train_images[validation_index]),
    'val_labels': jnp.asarray(train_labels[validation_index]),
    'test_images': jnp.asarray(test_images),
    'test_labels': jnp.asarray(test_labels),
    'n_train': int(train_index.shape[0]),
    'n_validation': int(validation_index.shape[0]),
    'n_test': int(test_images.shape[0]),
  }


def make_kernels(graphdef, state, tx):
  """One jitted epoch (a ``scan`` over the epoch's minibatch indices) and one jitted scoring pass."""

  def batch_loss(params, images, labels, index):
    model = nnx.merge(graphdef, params, state)
    logits = model(images[index])
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels[index]), logits

  @jax.jit
  def train_epoch(params, opt_state, images, labels, indices):

    def body(carry, index):
      params, opt_state = carry

      def objective(p):
        return jnp.mean(batch_loss(p, images, labels, index)[0])

      loss, grads = jax.value_and_grad(objective)(params)
      updates, opt_state = tx.update(grads, opt_state, params)
      return (optax.apply_updates(params, updates), opt_state), loss

    (params, opt_state), losses = jax.lax.scan(body, (params, opt_state), indices)
    return params, opt_state, losses

  @jax.jit
  def score(params, images, labels, index):
    losses, logits = batch_loss(params, images, labels, index)
    correct = (jnp.argmax(logits, axis=-1) == labels[index]).astype(jnp.float32)
    return losses, correct

  return train_epoch, score


def epoch_indices(rng, window, steps, batch):
  """``steps x batch`` draws from ``range(window)``, laid out as reshuffled passes over the window so
  a step never repeats an image and the window is covered evenly."""
  needed = steps * batch
  chunks = []
  total = 0
  while total < needed:
    chunks.append(rng.permutation(window))
    total += window
  return np.concatenate(chunks)[:needed].reshape(steps, batch).astype(np.int32)


def evaluate(score, params, images, labels, index, capacity, chunk):
  """Mean loss, its SEM and top-1 accuracy over ``index``.

  The index is padded to a whole number of ``chunk``-sized scoring passes and the loss vector is
  padded to a fixed ``capacity``, so every shape entering the kernels and
  :func:`~detopt.utils.training.masked_mean_sem` is constant as the growth window changes.
  """
  n = int(index.shape[0])
  n_chunks = (n + chunk - 1) // chunk
  padded_index = np.zeros(n_chunks * chunk, dtype=np.int32)
  padded_index[:n] = index
  losses = np.empty(padded_index.shape[0], dtype=np.float64)
  correct = np.empty(padded_index.shape[0], dtype=np.float64)
  for start in range(0, padded_index.shape[0], chunk):
    chunk_losses, chunk_correct = score(params, images, labels, jnp.asarray(padded_index[start:start + chunk]))
    losses[start:start + chunk] = np.asarray(chunk_losses, dtype=np.float64)
    correct[start:start + chunk] = np.asarray(chunk_correct, dtype=np.float64)
  buffer = np.zeros(capacity, dtype=np.float32)
  buffer[:n] = losses[:n]
  mean, sem = masked_mean_sem(jnp.asarray(buffer), n)
  return float(mean), float(sem), float(np.mean(correct[:n]))


def deviation_norm(params, initial):
  """L2 norm of ``params - initial`` over the whole parameter tree, the rewind's own audit: after a
  rewind of strength ``lambda`` this must fall by exactly ``1 - lambda``."""
  leaves = zip(jax.tree.leaves(params), jax.tree.leaves(initial))
  return float(jnp.sqrt(sum(jnp.sum(jnp.square(p - i)) for p, i in leaves)))


def train(mode, data, params0, graphdef, state, kernels, settings):
  """Run one arm to its own exit test and return its history.

  ``mode`` is ``'baseline'`` (full window, plateau exit) or ``'growth'`` (the design procedure with
  the exhausted-pool fallback). Everything else -- optimiser, epoch length, warmup, patience,
  precision, initial parameters -- is shared. ``settings['rewind']`` is the rewind strength applied
  at every successful data addition; at 0.0 the network and its optimiser state simply carry.
  """
  train_epoch, score = kernels
  batch = settings['batch']
  steps = settings['steps_per_epoch']
  patience = settings['patience']
  warmup = settings['warmup_epochs']
  precision = settings['loss_precision']
  n_train = data['n_train']
  chunk = settings['eval_chunk']
  rewind = float(settings['rewind'])

  tx = settings['tx']
  params = jax.tree.map(lambda x: x, params0)
  opt_state = tx.init(params)
  rng = np.random.default_rng(settings['shuffle_seed'])

  window = n_train if mode == 'baseline' else min(settings['n0'], n_train)
  validation_index = np.arange(data['n_validation'], dtype=np.int32)

  train_loss, val_loss, train_sem, val_sem, window_history = [], [], [], [], []
  growth_epochs = []
  round_start = 0
  fallback = False
  converged = False
  exit_test = None
  started = time.time()

  for epoch in range(1, settings['max_epochs'] + 1):
    indices = jnp.asarray(epoch_indices(rng, window, steps, batch))
    params, opt_state, _ = train_epoch(params, opt_state, data['train_images'], data['train_labels'], indices)
    train_mean, train_error, _ = evaluate(
      score, params, data['train_images'], data['train_labels'], np.arange(window, dtype=np.int32), n_train, chunk
    )
    val_mean, val_error, val_accuracy = evaluate(
      score, params, data['val_images'], data['val_labels'], validation_index, data['n_validation'], chunk
    )
    train_loss.append(train_mean)
    val_loss.append(val_mean)
    train_sem.append(train_error)
    val_sem.append(val_error)
    window_history.append(int(window))
    print(
      f"  [{mode}] epoch={epoch} window={window} train={train_mean:.4f}+-{train_error:.4f} "
      f"val={val_mean:.4f}+-{val_error:.4f} val_acc={val_accuracy:.4f} t={time.time() - started:.0f}s",
      flush=True
    )

    difference = abs(val_mean - train_mean)
    error = float(np.hypot(train_error, val_error))

    if epoch - round_start <= warmup:
      continue
    first = round_start + warmup
    tr = np.asarray(train_loss[first:], dtype=np.float64)
    va = np.asarray(val_loss[first:], dtype=np.float64)
    tr_s = np.asarray(train_sem[first:], dtype=np.float64)
    va_s = np.asarray(val_sem[first:], dtype=np.float64)
    if tr.shape[0] < 3:
      continue
    prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
    gap_sem = np.hypot(tr_s, va_s)
    gap_series = np.abs(va - tr) + gap_sem
    tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
    gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)

    if mode == 'baseline' or fallback:
      settled = probability_change_below(tr_mean, tr_cov, patience, 0.5 * precision)
      if settled > 0.9:
        converged = True
        exit_test = 'plateau' if mode == 'baseline' else 'plateau-fallback'
        print(
          f"  [{mode}/plateau] train={train_mean:.4f} val={val_mean:.4f} diff={difference:.4f} err={error:.4f} "
          f"| P(settled)={settled:.3f} | window={window} epochs={epoch}",
          flush=True
        )
        break
      continue

    add_data = False
    p_gap_exceeds = probability_above(gap_mean, gap_cov, patience, precision, gap_series.shape[0])
    if p_gap_exceeds > 0.9:
      add_data = True
    else:
      settled = probability_change_below(tr_mean, tr_cov, patience, 0.5 * precision)
      if settled > 0.9:
        if difference + error > precision:
          add_data = True
        else:
          converged = True
          exit_test = 'bayes-converged'
          print(
            f"  [{mode}/converged] train={train_mean:.4f} val={val_mean:.4f} diff={difference:.4f} err={error:.4f} "
            f"prec={precision:.4f} | P(gap>LP)={p_gap_exceeds:.3f} P(settled)={settled:.3f} | window={window}",
            flush=True
          )
          break
      else:
        continue

    if add_data and window >= n_train:
      fallback = True
      settled = probability_change_below(tr_mean, tr_cov, patience, 0.5 * precision)
      print(f"  [{mode}/pool-exhausted] window={window} -> plateau fallback at epoch {epoch}", flush=True)
      if settled > 0.9:
        converged = True
        exit_test = 'plateau-fallback'
        break
      continue
    if add_data:
      window = min(window + settings['n_increment'], n_train)
      if rewind > 0.0:
        before = deviation_norm(params, params0)
        params = jax.tree.map(lambda current, initial: initial + (1.0 - rewind) * (current - initial), params, params0)
        opt_state = tx.init(params)
        after = deviation_norm(params, params0)
        print(
          f"  [{mode}/rewind] lambda={rewind:.3f} epoch={epoch} window={window} "
          f"|params-initial| {before:.4f} -> {after:.4f} ratio={after / max(before, 1.0e-12):.4f}",
          flush=True
        )
      round_start = epoch
      growth_epochs.append(epoch)

  elapsed = time.time() - started
  test_index = np.arange(data['n_test'], dtype=np.int32)
  test_mean, test_sem, test_accuracy = evaluate(
    score, params, data['test_images'], data['test_labels'], test_index, data['n_test'], chunk
  )
  return {
    'mode': mode,
    'converged': bool(converged),
    'exit_test': exit_test,
    'epochs': len(train_loss),
    'steps': len(train_loss) * steps,
    'images_consumed': int(window_history[-1]),
    'final_train_loss': float(train_loss[-1]),
    'final_train_sem': float(train_sem[-1]),
    'final_val_loss': float(val_loss[-1]),
    'final_val_sem': float(val_sem[-1]),
    'test_accuracy': float(test_accuracy),
    'test_cross_entropy': float(test_mean),
    'test_cross_entropy_sem': float(test_sem),
    'wall_clock_seconds': float(elapsed),
    'train_loss_per_epoch': [float(x) for x in train_loss],
    'val_loss_per_epoch': [float(x) for x in val_loss],
    'train_sem_per_epoch': [float(x) for x in train_sem],
    'val_sem_per_epoch': [float(x) for x in val_sem],
    'window_per_epoch': [int(x) for x in window_history],
    'data_added_at_epochs': [int(e) for e in growth_epochs],
  }


def snapshot(history, n_validation):
  """The trainers' own epoch-snapshot dict, which :func:`plot_iteration` consumes."""
  return {
    'train_loss_per_epoch': np.asarray(history['train_loss_per_epoch'], dtype=np.float64),
    'val_loss_per_epoch': np.asarray(history['val_loss_per_epoch'], dtype=np.float64),
    'train_sem_per_epoch': np.asarray(history['train_sem_per_epoch'], dtype=np.float64),
    'val_sem_per_epoch': np.asarray(history['val_sem_per_epoch'], dtype=np.float64),
    'train_budget_per_epoch': np.asarray(history['window_per_epoch'], dtype=np.float64),
    'final_train_budget': int(history['window_per_epoch'][-1]),
    'final_val_pool_size': int(n_validation),
  }


def plot_overlay(histories, path):
  """Every arm's train and validation curves against epoch and against images consumed."""
  from matplotlib.figure import Figure

  fig = Figure(figsize=(14, 5.5))
  left, right = fig.subplots(1, 2)
  for position, name in enumerate(sorted(histories)):
    history = histories[name]
    color = ARM_COLORS.get(name, FALLBACK_COLORS[position % len(FALLBACK_COLORS)])
    epochs = np.arange(1, history['epochs'] + 1)
    tl = np.asarray(history['train_loss_per_epoch'])
    vl = np.asarray(history['val_loss_per_epoch'])
    window = np.asarray(history['window_per_epoch'], dtype=np.float64)
    left.plot(epochs, tl, color=color, ls='-', lw=1.2, label=f'{name} train')
    left.plot(epochs, vl, color=color, ls='--', lw=1.2, label=f'{name} validation')
    right.plot(window, tl, color=color, ls='-', lw=1.0, marker='.', ms=2, label=f'{name} train')
    right.plot(window, vl, color=color, ls='--', lw=1.0, marker='.', ms=2, label=f'{name} validation')
    for epoch in history['data_added_at_epochs']:
      left.axvline(epoch, color=color, ls=':', alpha=0.16, lw=0.6)
  left.set_xlabel('epoch')
  left.set_ylabel('softmax cross-entropy')
  left.set_yscale('log')
  left.set_title('loss vs epoch (dotted: each arm\'s data additions)')
  left.grid(True, alpha=0.3)
  left.legend(loc='lower left', fontsize=8)
  right.set_xlabel('images consumed (training window)')
  right.set_ylabel('softmax cross-entropy')
  right.set_yscale('log')
  right.set_xscale('log')
  right.set_title('loss vs images consumed')
  right.grid(True, alpha=0.3)
  right.legend(loc='lower left', fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=120)
  return path


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--data', default='data/cifar10/cifar10.npz')
  parser.add_argument('--output', default='output/cifar-growth')
  parser.add_argument('--split-seed', type=int, default=20260823)
  parser.add_argument('--init-seed', type=int, default=770315)
  parser.add_argument('--shuffle-seed', type=int, default=4242)
  parser.add_argument('--n-validation', type=int, default=10000)
  parser.add_argument('--batch', type=int, default=128)
  parser.add_argument('--n0', type=int, default=2048)
  parser.add_argument('--n-increment', type=int, default=1024)
  parser.add_argument('--loss-precision', type=float, default=0.05)
  parser.add_argument('--warmup-epochs', type=int, default=4)
  parser.add_argument('--patience', type=int, default=16)
  parser.add_argument('--max-epochs', type=int, default=800)
  parser.add_argument('--learning-rate', type=float, default=5.0e-4)
  parser.add_argument('--weight-decay', type=float, default=1.0e-3)
  parser.add_argument('--eval-chunk', type=int, default=1000)
  parser.add_argument('--param-mix', type=float, default=0.0)
  parser.add_argument('--arms', default='baseline,growth')
  parser.add_argument('--run-name', default='')
  arguments = parser.parse_args()

  data = load_split(arguments.data, arguments.split_seed, arguments.n_validation)
  steps_per_epoch = data['n_train'] // arguments.batch
  print(
    f"split seed={arguments.split_seed} train={data['n_train']} validation={data['n_validation']} "
    f"test={data['n_test']} | epoch={steps_per_epoch} steps of batch {arguments.batch}",
    flush=True
  )

  model = ConvNet(rngs=nnx.Rngs(params=arguments.init_seed))
  graphdef, params0, state = nnx.split(model, nnx.Param, nnx.Variable)
  n_parameters = int(sum(np.prod(leaf.shape) for leaf in jax.tree.leaves(params0)))
  print(f"init seed={arguments.init_seed} parameters={n_parameters}", flush=True)

  tx = optax.adamaxw(
    learning_rate=arguments.learning_rate, b1=0.9, b2=0.999, eps=1.0e-8, weight_decay=arguments.weight_decay
  )
  kernels = make_kernels(graphdef, state, tx)
  settings = {
    'batch': arguments.batch,
    'steps_per_epoch': steps_per_epoch,
    'patience': arguments.patience,
    'warmup_epochs': arguments.warmup_epochs,
    'loss_precision': arguments.loss_precision,
    'max_epochs': arguments.max_epochs,
    'n0': arguments.n0,
    'n_increment': arguments.n_increment,
    'shuffle_seed': arguments.shuffle_seed,
    'eval_chunk': arguments.eval_chunk,
    'rewind': arguments.rewind,
    'tx': tx,
  }

  common = {
    'split_seed': arguments.split_seed,
    'init_seed': arguments.init_seed,
    'shuffle_seed': arguments.shuffle_seed,
    'n_train': data['n_train'],
    'n_validation': data['n_validation'],
    'n_test': data['n_test'],
    'batch': arguments.batch,
    'steps_per_epoch': steps_per_epoch,
    'loss_precision': arguments.loss_precision,
    'warmup_epochs': arguments.warmup_epochs,
    'patience': arguments.patience,
    'max_epochs': arguments.max_epochs,
    'n0': arguments.n0,
    'n_increment': arguments.n_increment,
    'rewind': arguments.rewind,
    'channels': list(CHANNELS),
    'activation': 'relu',
    'n_parameters': n_parameters,
    'optimiser': {
      'name': 'adamaxw',
      'learning_rate': arguments.learning_rate,
      'b1': 0.9,
      'b2': 0.999,
      'eps': 1.0e-8,
      'weight_decay': arguments.weight_decay,
      'schedule': 'constant',
    },
  }

  requested = [name.strip() for name in arguments.arms.split(',') if len(name.strip()) > 0]
  if len(arguments.run_name) > 0 and len(requested) != 1:
    raise ValueError(f'--run-name names ONE output directory, so it needs exactly one arm; got {requested}')
  for mode in requested:
    name = arguments.run_name if len(arguments.run_name) > 0 else mode
    print(f"=== {name} (mode={mode}, rewind={arguments.rewind}) ===", flush=True)
    history = train(mode, data, params0, graphdef, state, kernels, settings)
    run_directory = os.path.join(arguments.output, name)
    os.makedirs(run_directory, exist_ok=True)
    record = dict(common)
    record.update(history)
    record['run_name'] = name
    with open(os.path.join(run_directory, 'result.json'), 'w') as handle:
      json.dump(record, handle, indent=2)
    plot_iteration(
      snapshot(history, data['n_validation']), 0, {
        'run': name,
        'mode': mode,
        'rewind': arguments.rewind,
        'split_seed': arguments.split_seed,
        'init_seed': arguments.init_seed,
        'window_start': history['window_per_epoch'][0],
        'window_final': history['window_per_epoch'][-1],
      }, history['final_val_loss'], run_directory
    )
    print(
      f"[{name}] converged={history['converged']} by {history['exit_test']} | epochs={history['epochs']} "
      f"steps={history['steps']} images={history['images_consumed']} | train={history['final_train_loss']:.4f} "
      f"val={history['final_val_loss']:.4f} | test_acc={history['test_accuracy']:.4f} "
      f"test_ce={history['test_cross_entropy']:.4f} | {history['wall_clock_seconds']:.0f}s",
      flush=True
    )

  # The overlay is rebuilt from EVERY arm on disk, so an invocation that ran one arm still redraws
  # the full comparison without re-running -- or overwriting -- the arms it did not touch.
  histories = {}
  for entry in sorted(os.listdir(arguments.output)):
    candidate = os.path.join(arguments.output, entry, 'result.json')
    if os.path.isfile(candidate):
      with open(candidate) as handle:
        histories[entry] = json.load(handle)
  print(f"overlay arms: {sorted(histories)}", flush=True)
  plot_overlay(histories, os.path.join(arguments.output, 'overlay.png'))
  print('done', flush=True)


if __name__ == '__main__':
  main()
