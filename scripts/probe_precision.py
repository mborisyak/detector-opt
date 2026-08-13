#!/usr/bin/env python3
"""What `loss_precision` can this task ACTUALLY be run at? -- `docs/tuning-preregistration.md` section 5.

    python scripts/probe_precision.py =enzyme_extremes --seeds 1 126382657 \
        --output output/screen/precision-extremes.json

THE PROTOCOL, which is binding and is not this script's invention (pre-registration section 5):
measure what the trainer RELIABLY reaches across >= 8 DESIGNS x >= 2 SEEDS, take the MAXIMUM, fix it,
and only THEN ask whether the task passes with it. A single-design estimate is insufficient -- a
sibling task's floor ran 0.0055 to 0.0083 across designs and the BINDING design was the bad one.

WHAT IS MEASURED. `detopt/nn/trainer/design.py` calls a design converged when `diff + err` falls under
`loss_precision`, where `err = hypot(train_sem, val_sem)` is the loss estimate's standard error and
`diff = |val - train|` is the train/validation gap. `err` falls like `1/sqrt(window)`, so more data
always fixes it; `diff` is an OVERFITTING BIAS and need not fall at all. A design whose persistent gap
exceeds the requested precision can NEVER converge, grows its window to `iteration_limit`, and
`scripts/bo.py` raises -- deterministically per seed, so a resubmission reproduces it.

WHICH DESIGNS, and this is the trap the protocol exists to avoid: **the binding designs are the
INFORMATIVE ones**. On this classification task an uninformative design leaves both train and val at
the no-information level 1.0, so its gap is ~0; it is the design the network can actually FIT that
overfits, and the gap opens there. Eight uniform random draws sit near the ceiling and converge
trivially, which is why `--designs` defaults to a set that SPANS the landscape: the proxy's best,
several of its quantiles, its worst, and the physically-motivated box corners from
`scripts/validate_inhibitor.py` (the 2x2 factorial the chemistry implies, and the ceiling probe).

RUN AT THE CONFIG'S OWN PRECISION. `loss_precision` also enters `is_plateaued` and the large-gap rule,
so a run at a different precision follows a DIFFERENT trajectory and is not comparable. Everything is
therefore scored at whatever the run config declares, and the consequence is stated plainly in the
output: a design that converges is CENSORED at that value (it proves `slack <= precision`, not what
its floor is), and only a design that hits the cap reports an uncensored number. That is exactly the
quantity the deliverable needs -- the maximum -- because the cappers are what set it.

REPORTED per (design, seed, setting): `diff`, `err`, `diff + err`, the achieved train/validation
losses and the objective `(train + val) / 2`, the window reached, whether it hit the cap, the detector
calls spent and the wall clock. The last round's `diff` mean/max are reported alongside, because the
`diff` at the stopping epoch is a STOPPED -- hence downward-biased -- sample of the gap.

SWEEPS. `--weight-decay` and `--iteration-limit` are the two levers on the binding term. Both are
reported WITH the achieved validation loss, never alone: a setting that closes the gap by
underfitting BOTH sides has bought precision by destroying the signal the benchmark is made of, and
only the level says which happened. `--plot-per-epoch 0 1` measures what the per-epoch diagnostic
figure costs in wall clock, on the same design and seed.

This script SETS NOTHING. It writes a JSON of measurements; fixing `loss_precision` is a
pre-registration act made in a document, not here.
"""

import argparse
import itertools
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("AGG")

import numpy as np
import scipy.stats

import detopt
import detopt.detector
from detopt.nn.trainer import DesignTrainer
from detopt.utils.viz.bo import plot_iteration

# The per-window collapse of the epoch history is already written and already documents why the
# stopping epoch's `diff` understates the gap -- reuse it rather than keep a second copy.
from probe_dropout import _per_window
from validate_inhibitor import named_designs


def design_set(detector, landscape, names=None, replay=(), extra_ranks=(), unranked=0):
  """The designs to score, as SCALED `[0, 1]^d` vectors, keyed by name.

  Two sources, because neither alone spans what the protocol asks for:

  * the task's own Sobol landscape (`landscape`, a `_m<m>.npz` written by `scripts/screen_task.py`),
    taken at fixed ORDER STATISTICS of the proxy loss -- best, 2%, 10%, median, 90%, worst. The median
    is the criterion's own baseline (the median random design), and the best is where the network has
    the most to fit and therefore the most to overfit;
  * the named designs of `scripts/validate_inhibitor.py` -- the 2x2 factorial the chemistry implies
    (a genuine CORNER of the box in the two concentration coordinates, and the informative one), the
    same crossing at the top of the inhibitor box, and the deliberately uninformative ceiling probe
    (the opposite corner). These are physics, not order statistics, so they do not move if the
    landscape is re-measured.

  The landscape is used ONLY to CHOOSE designs. Every number this script reports is measured fresh
  through the neural trainer, so a stale proxy file costs a less well-spread sample and nothing else.
  """
  chosen = {}
  # UNRANKED designs, for batch sizes that have no landscape. A landscape costs one proxy fit per
  # design and the proxy is the flattened-read-out estimator, which is exactly what is not trusted at
  # large `n_experiments`; `unranked` therefore draws designs from the scaled cube directly and names
  # them by construction, never by a proxy loss it does not have. `centre` and `corner-low` are fixed
  # points of the box, the rest a Sobol sample.
  if unranked > 0:
    dimension = int(detector.design_dim())
    chosen["centre"] = (np.full(dimension, 0.5, np.float32), float("nan"))
    chosen["corner-low"] = (np.zeros(dimension, np.float32), float("nan"))
    points = scipy.stats.qmc.Sobol(dimension, scramble=True, seed=0).random(int(unranked))
    for index, point in enumerate(points):
      chosen[f"unranked-{index}"] = (np.asarray(point, np.float32), float("nan"))
  if landscape is not None and len(str(landscape)) > 0:
    with np.load(landscape) as data:
      designs, losses = np.asarray(data["designs"], np.float32), np.asarray(data["losses"], np.float64)
    if designs.shape[1] != int(detector.design_dim()):
      # A landscape from a different batch size cannot name a design for this one. Say so here rather
      # than let a wrong-length vector reach the trainer.
      raise SystemExit(f"probe_precision: {landscape} holds {designs.shape[1]}-dimensional designs but "
                       f"the detector's design is {int(detector.design_dim())}-dimensional")
    order = np.argsort(losses)  # ascending loss: best first
    n = len(order)
    for label, rank in (("sobol-best", 0), ("sobol-p02", round(0.02 * n)), ("sobol-p10", round(0.10 * n)),
                        ("sobol-median", n // 2), ("sobol-p90", round(0.90 * n)), ("sobol-worst", n - 1)):
      index = int(order[min(rank, n - 1)])
      chosen[label] = (np.asarray(designs[index], np.float32), float(losses[index]))
    # EXTRA RANKS, the GOOD END. Criterion (d) tests `best@n - best@2n`, and both terms are best-so-far
    # losses within one run -- so the pair it differences is two GOOD designs, never a good one against
    # the landscape's worst. The six order statistics above put only three designs below the proxy's 10th
    # percentile, i.e. three pairs, which is not a distribution. Requesting further ranks from the same
    # ascending order fills that end in without disturbing anything already measured.
    for rank in extra_ranks:
      index = int(order[min(int(rank), n - 1)])
      chosen[f"sobol-rank{int(rank)}"] = (np.asarray(designs[index], np.float32), float(losses[index]))
  # The physics-stated calibration designs exist only for the INHIBITOR chemistry -- `named_designs`
  # is built out of `inhibitor_bounds` and `inhibitor_potency_bounds`, which the Michaelis-Menten
  # detector does not have. Offer them where the detector supplies what they are made of, and leave
  # them out otherwise; asking for one by name on a detector that lacks it still fails loudly through
  # the "unknown design(s)" check below.
  if all(hasattr(detector, field) for field in ("inhibitor_bounds", "inhibitor_potency_bounds", "substrate_B_bounds")):
    named = named_designs(detector)
    for label, key in (("corner-2x2", "KNOWN-GOOD 2x2, two inhibitor doses"),
                       ("corner-2x2-strong", "KNOWN-GOOD 2x2, strong dose"),
                       ("corner-uninformative", "uninformative (ceiling probe)")):
      chosen[label] = (np.asarray(detector.to_scaled(named[key]), np.float32), float("nan"))
  # The literal upper corner of the scaled cube: every experiment at the top of every range, so no
  # contrast anywhere. Its partner (the lower corner) is `corner-uninformative` above.
  chosen["corner-high"] = (np.ones(int(detector.design_dim()), np.float32), float("nan"))
  for label, x_scaled in replay:
    chosen[label] = (np.asarray(x_scaled, np.float32), float("nan"))
  if names is not None and len(names) > 0:
    missing = [name for name in names if name not in chosen]
    if len(missing) > 0:
      raise SystemExit(f"probe_precision: unknown design(s) {missing}; have {sorted(chosen)}")
    chosen = {name: chosen[name] for name in names}
  return chosen


def replay_design(detector, config, results_path, seed):
  """The design a crashed `scripts/bo.py` run would have scored NEXT -- i.e. the one that killed it.

  `bo.py` writes `results.json` only for COMPLETED iterations, so the design that raised is not in
  the file. It is recoverable anyway, because `BayesianOptimizer` is deterministic given its seed and
  its observations: the initial block is a scrambled Sobol sequence keyed by the seed alone, and every
  later proposal is an EI maximisation over a GP fitted with `random_state = seed`. Replaying the
  recorded `(x_scaled, loss, loss_std)` triples in order and proposing once more therefore reproduces
  it -- and replaying VERIFIES itself, because each recorded proposal must come back before it is
  appended. The mismatches are returned rather than raised, so a design is never silently wrong.

  These are the designs the protocol most wants: not a quantile of a landscape but the exact points
  that made two campaign runs die.
  """
  import detopt.bo
  from detopt.bo import BayesianOptimizer

  with open(results_path) as f:
    recorded = json.load(f)["results"]
  gp_cfg, ei_cfg = dict(config["bo"]["gp"]), dict(config["bo"]["ei"])
  n_init = int(config["bo"].get("n_init", gp_cfg["n_folds"]))
  kernel = detopt.bo.kernel_from_config(gp_cfg.pop("kernel"), detector, gp_cfg)
  optimiser = BayesianOptimizer(int(detector.design_dim()), gp=gp_cfg, ei=ei_cfg, kernel=kernel,
                                n_init=n_init, seed=int(seed))
  mismatch = []
  for iteration, entry in enumerate(recorded):
    proposed = np.asarray(optimiser.propose(), np.float32)
    x = np.asarray(entry["x_scaled"], np.float32)
    deviation = float(np.max(np.abs(proposed - x)))
    if deviation > 1e-5:
      mismatch.append((iteration, deviation))
    optimiser.append(x, float(entry["loss"]), noise=float(entry["loss_std"]))
  return np.asarray(optimiser.propose(), np.float32), len(recorded), mismatch


def run_config_for(config, *, weight_decay, iteration_limit, precision, device, n0=None,
                   n_increment=None, warmup_epochs=None, patience=None, n_models=None,
                   p_dropout=None, reinit_on_grow=None, param_mix=None, plot_per_epoch=None):
  """A deep copy of the run config with this cell's overrides applied (nothing is written to disk).

  The budget pool is sized to TWICE one design's window rather than to the campaign's whole budget:
  this script scores one design per trainer, so the campaign budget would only preallocate ~450 MB of
  GPU pool that is never filled -- on a card that is shared between two SLURM shards. It cannot change
  what is measured (`_sample_round` reports a full WINDOW and a full POOL differently, and the run is
  recorded as an error if the pool is what ran out).

  `n0 = iteration_limit` removes the growth loop, so every design is scored at the SAME window, and
  holding the product fixed keeps the campaign's own definition of "trained out" while the
  convergence bar moves.
  """
  run = json.loads(json.dumps(config))  # plain JSON-able yaml
  training = run["training"]
  if iteration_limit is not None:
    training["iteration_limit"] = int(iteration_limit)
  if precision is not None:
    training["loss_precision"] = float(precision)
  if n0 is not None:
    training["n0"] = int(training["iteration_limit"]) if int(n0) < 0 else int(n0)
  # The DATA SCHEDULE, the other way at the gap. `n0` and `n_increment` set how many times the earliest
  # events are re-exposed (the first `n0` sit in every stage, the last arrive only at the end);
  # `warmup_epochs` and `patience` set how much room the plateau test has before it may stop.
  if n_increment is not None:
    training["n_increment"] = int(n_increment)
  if warmup_epochs is not None:
    training["warmup_epochs"] = int(warmup_epochs)
  if patience is not None:
    training["patience"] = int(patience)
  # The RULE rather than the schedule: gate rule 1 on the gap having stopped falling. Left UNSET the
  # key is absent and the trainer's own default (False = the original rule) applies, so a run that does
  # not ask for it is unaffected by the option existing.
  # THE NETWORK at the data addition rather than the schedule of additions. Left UNSET the keys are
  # absent and the trainer's own defaults (carry the params AND the optimiser state, unperturbed)
  # apply, so a run that does not ask for either is unaffected by the options existing.
  if reinit_on_grow is not None:
    training["reinit_on_grow"] = bool(reinit_on_grow)
  if param_mix is not None:
    training["param_mix"] = float(param_mix)
  if plot_per_epoch is not None:
    run["plot_per_epoch"] = bool(int(plot_per_epoch))
  if device is not None:
    run["device"] = device
  training["budget"] = int(2 * training["iteration_limit"] / (1.0 - float(training["val_fraction"])))
  if weight_decay is not None:
    (optimizer_name, ), = (training["optimizer"].keys(), )
    training["optimizer"][optimizer_name]["weight_decay"] = float(weight_decay)
  # The REGRESSOR knobs. `n_models = 1` is a one-member ensemble, i.e. the ensemble off: averaging
  # members suppresses variance without touching the bias shared by all of them, which hides what the
  # train/validation gap is doing.
  (regressor_name, ), = (run["regressor"].keys(), )
  if n_models is not None:
    run["regressor"][regressor_name]["n_models"] = int(n_models)
  if p_dropout is not None:
    run["regressor"][regressor_name]["p_dropout"] = float(p_dropout)
  return run


def init_signature(trainer, seed):
  """A checksum of the INITIAL network, before a single training step.

  Two arms compared cell by cell must start from byte-identical parameters, or the difference between
  them carries initialisation noise as well as the effect under test -- and that noise has been
  measured at 0.003-0.009 on this task, the size of the effects being reported.

  `DesignTrainer.train` derives its init from the `SeedSequence` it is handed and from NOTHING else
  (`init_seq, training_seq = seed_seq.spawn(2)`, then `fresh_design_network`), so rebuilding that
  sequence here reproduces exactly the network the run is about to start from. It cannot disturb the
  run: `generate_state` is pure, and this spawns from its own fresh copy of the sequence rather than
  from the object handed to the trainer.
  """
  import jax
  import jax.numpy as jnp

  from detopt.nn.trainer.common import fresh_design_network

  init_seq, _ = np.random.SeedSequence(int(seed)).spawn(1)[0].spawn(2)
  params, _, _ = fresh_design_network(trainer, init_seq, None)
  leaves = jax.tree.leaves(params)
  checksum = float(sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves))
  return checksum, int(sum(int(leaf.size) for leaf in leaves))


def measure(detector, run, x_scaled, seed, *, plot_dir=None):
  """Score ONE (design, seed, setting) through the trainer and return what it achieved.

  The per-epoch history is the primary source for `diff` / `err`, because it is available on BOTH
  exits: the trainer submits the epoch's snapshot to `on_epoch` before it decides, so the last row is
  the converged epoch on a convergence and the capped epoch on a `RuntimeError`. `objective_std` is
  cross-checked against it where the design converged.
  """
  trainer = DesignTrainer.from_config(detector, run, checkpoint_dir=None, seed=seed)
  init_checksum, n_parameters = init_signature(trainer, seed)
  print(f"  init |params|_1 = {init_checksum:.6f} over {n_parameters} parameters "
        f"(must MATCH cell-for-cell across arms being differenced)", flush=True)
  history = {}
  if plot_dir is None:
    on_epoch = history.update
  else:  # the Task-0 A/B: exactly what scripts/bo.py used to do unconditionally, per epoch
    os.makedirs(plot_dir, exist_ok=True)

    def on_epoch(snapshot):
      history.update(snapshot)
      values = snapshot["val_loss_per_epoch"]
      live = float(values[-1]) if values.size > 0 else float("nan")
      plot_iteration(snapshot, iteration=0, design=np.asarray(x_scaled).tolist(), val_loss=live, plots_dir=plot_dir)

  started = time.time()
  status, message = "converged", ""
  objective, objective_std, spent = float("nan"), float("nan"), -1
  try:
    result = trainer.train(x_scaled, np.random.SeedSequence(int(seed)).spawn(1)[0], step=0, on_epoch=on_epoch)
    if result is None:
      status, message = "pool exhausted", "the shared budget pool filled -- this cell is NOT a measurement"
    else:
      objective, objective_std, spent = float(result.objective_loss), float(result.objective_std), int(result.spent)
  except RuntimeError as error:
    # ONLY the trainer's own window-cap error. A JAX runtime failure (an OOM, say) also subclasses
    # RuntimeError, and recording one as "the design could not reach the precision" would invent a
    # measurement out of an infrastructure fault -- so anything else is re-raised.
    text = str(error).replace("\n", " ")
    if "did not reach precision within iteration_limit" not in text:
      raise
    status, message = "capped", text
  wall = time.time() - started

  row = {"status": status, "message": message, "wall_s": wall, "spent": spent,
         "objective": objective, "objective_std": objective_std,
         "init_checksum": init_checksum, "n_parameters": n_parameters}
  per_window = _per_window(history)
  if len(per_window) > 0:
    last = per_window[-1]
    row.update({
      "train": last["train"], "val": last["val"], "diff": last["diff"], "err": last["err"],
      "slack": last["diff"] + last["err"], "window": last["window"],
      "diff_last_round_mean": last["diff_mean"], "diff_last_round_max": last["diff_max"],
      "n_epochs": int(np.asarray(history["train_loss_per_epoch"]).size),
      "n_rounds": len(per_window),
    })
    row["per_window"] = per_window
  return row


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config", help="gearup root token of a RUN config, e.g. =enzyme_extremes")
  parser.add_argument("--landscape", default="output/screen/extremes_m4.npz",
                      help="a screen_task.py `_m<m>.npz` (designs + proxy losses), used ONLY to pick "
                           "designs spanning the landscape -- never as a reported number")
  parser.add_argument("--designs", nargs="*", default=None, help="subset by name (default: all of them)")
  parser.add_argument("--prior-scale", type=float, default=None, metavar="S",
                      help="MAP prior on the KERNELS: adds `S * model.regularization() / N` to the training "
                           "loss, where N is the current window. The model states -log p(kernel) under the "
                           "prior that maps a standard normal input to a standard normal output (variance "
                           "1/fan_in, the Lecun scale it is initialised at); S=1 is the MAP coefficient "
                           "exactly, S=0 (the default) is the objective as it was before this existed. The "
                           "REPORTED train/validation losses and the convergence test are unaffected -- the "
                           "prior enters the gradient only.")
  parser.add_argument("--unranked", type=int, default=0, metavar="N",
                      help="add N Sobol designs plus `centre` and `corner-low`, named by CONSTRUCTION "
                           "rather than by a proxy loss. For batch sizes that have no landscape -- pass "
                           "--landscape '' to work without one entirely.")
  parser.add_argument("--extra-ranks", type=int, nargs="*", default=[], metavar="RANK",
                      help="further landscape designs by ASCENDING-LOSS rank (0 = best), named "
                           "`sobol-rank<RANK>`. Use it to fill in the GOOD END: criterion (d) "
                           "differences two best-so-far losses, so the pairs it makes are good-against-"
                           "good, and the six default order statistics leave too few designs there to "
                           "form a pairwise distribution")
  parser.add_argument("--replay", nargs="*", default=[], metavar="RESULTS.JSON:SEED",
                      help="add the design a crashed bo.py run would have scored NEXT -- the one that "
                           "killed it, recovered by replaying its recorded proposals (see replay_design). "
                           "Named `crash-s<SEED>`. These are the binding designs by construction")
  parser.add_argument("--seeds", type=int, nargs="+", default=[1, 126382657],
                      help="the protocol requires >= 2; the campaign's own seeds make it paired with a run")
  parser.add_argument("--weight-decay", type=float, nargs="+", default=[None],
                      help="AdamW weight decay to sweep (default: whatever the config declares)")
  parser.add_argument("--iteration-limit", type=int, nargs="+", default=[None],
                      help="per-design window cap AND epoch length (default: the config's)")
  parser.add_argument("--precision", type=float, default=None,
                      help="override `training.loss_precision`. It also enters `is_plateaued` and the "
                           "large-gap rule, so a run at a different precision is a DIFFERENT trajectory "
                           "and must not be mixed into a table with the config's own value")
  parser.add_argument("--n0", type=int, default=None,
                      help="override `training.n0`; -1 means `= iteration_limit`, which removes the "
                           "growth loop entirely and scores EVERY design at the same window. That is "
                           "what makes a cell UNCENSORED: at the config's own precision a design that "
                           "converges only proves `slack <= precision`, so the table is a set of "
                           "upper bounds pressed against the bar. One round at the window cap, with a "
                           "`loss_precision` loosened far above any plausible gap (and `--flatness-tol` "
                           "moved to keep their PRODUCT, hence `is_plateaued`, unchanged), instead "
                           "reports where each design actually settles")
  parser.add_argument("--n-increment", type=int, default=None,
                      help="override `training.n_increment`, the events added per growth round")
  parser.add_argument("--warmup-epochs", type=int, default=None,
                      help="override `training.warmup_epochs`, the unconditional epochs after every "
                           "data addition before the stopping rule may act")
  parser.add_argument("--patience", type=int, default=None,
                      help="override `training.patience`, the epochs the plateau test fits its slope over")
  parser.add_argument("--n-models", type=int, default=None,
                      help="override the regressor's `n_models`. `1` is a ONE-MEMBER ensemble, i.e. the "
                           "ensemble off: averaging members suppresses the variance between them but "
                           "not the bias they share, so it hides what the train/validation gap is doing")
  parser.add_argument("--p-dropout", type=float, default=None,
                      help="override the regressor's `p_dropout` (0 = dropout off)")
  parser.add_argument("--reinit-on-grow", action="store_true", default=None,
                      help="REBUILD the network from scratch -- params, buffer state and optimiser "
                           "state -- at every data addition, holding the growth schedule fixed. The "
                           "schedule sweeps hold the network and vary WHEN data arrives; this varies "
                           "the network and holds the schedule, so between them the two confounded "
                           "differences separate. Unset leaves the trainer's default (carry it)")
  parser.add_argument("--param-mix", type=float, default=None, metavar="LAMBDA",
                      help="at every data addition set params to (1-LAMBDA)*current + LAMBDA*fresh_init, "
                           "keeping the Adam moments. 0 is the carried network, 1 is a fresh draw with "
                           "momentum kept -- i.e. --reinit-on-grow minus the optimiser reset.")
  parser.add_argument("--plot-per-epoch", type=int, nargs="+", default=[0], choices=[0, 1],
                      help="render the per-epoch diagnostic figure. Pass `0 1` to measure what it costs")
  parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                      help="dotted override on the DETECTOR config, the repo's usual form (as "
                           "`scripts/screen_task.py --set`), e.g. "
                           "`--set enzyme_inhib.n_measurements=32`. `loss_precision` is a property of "
                           "the OPERATING POINT -- `m` and the read-out count change both what the "
                           "network can fit and how much it can memorise -- so the cell measured is "
                           "pinned here rather than by editing a config")
  parser.add_argument("--device", default=None, help="override `device` (e.g. cpu)")
  parser.add_argument("--plots-dir", default="output/screen/precision-plots")
  parser.add_argument("--output", default="output/screen/precision-extremes.json")
  parser.add_argument("--resume", action="store_true",
                      help="keep cells already in --output and skip them (a SLURM job that was killed "
                           "should cost only what is missing)")
  arguments = parser.parse_args()

  import yaml  # local, so the module stays importable without a config
  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  # A RUN config names its detector by string and gearup resolves it against config/detector/<name>.yaml.
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  if len(arguments.overrides) > 0:
    import detopt.utils.config
    detector_config = detopt.utils.config.override(detector_config, list(arguments.overrides))
    print(f"detector overrides: {arguments.overrides}", flush=True)
  detector = detopt.detector.from_config(detector_config)

  # Recovered crash designs are a BONUS: if a replay fails, the protocol's own design set must still
  # be measured, so the failure is reported and the run continues.
  replay = []
  for spec in arguments.replay:
    path, _, seed = spec.rpartition(":")
    try:
      x_scaled, n_recorded, mismatch = replay_design(detector, config, path, int(seed))
      label = f"crash-s{int(seed)}"
      replay.append((label, x_scaled))
      status = "EXACT" if len(mismatch) == 0 else f"MISMATCH on {mismatch}"
      print(f"replay {path} seed {seed}: {n_recorded} recorded proposals reproduced {status}; "
            f"design {n_recorded + 1} recovered as `{label}`", flush=True)
    except Exception as error:  # noqa: BLE001 -- a bonus design must not take the deliverable with it
      print(f"replay {path} seed {seed} FAILED ({type(error).__name__}: {error}); continuing without it",
            flush=True)

  # A `crash-*` name is CONDITIONAL on its replay having worked, so asking for one that did not
  # arrive drops that name with a warning rather than killing the run; any other unknown name is a
  # typo and still stops the run (design_set).
  requested = arguments.designs
  if requested is not None:
    recovered = {label for label, _ in replay}
    dropped = [n for n in requested if n.startswith("crash-") and n not in recovered]
    if len(dropped) > 0:
      print(f"[warning] dropping {dropped}: their replay produced no design", flush=True)
      requested = [n for n in requested if n not in dropped]

  designs = design_set(detector, arguments.landscape, requested, replay, arguments.extra_ranks,
                       arguments.unranked)
  rows, done = [], set()
  if arguments.resume and os.path.isfile(arguments.output):
    with open(arguments.output) as f:
      rows = json.load(f)["rows"]
    # The key carries the SETTING as well as the cell: `n0` and `loss_precision` each change the
    # trajectory, so two rows that differ in either are different measurements, not a repeat.
    done = {(r["design"], r["seed"], r["weight_decay"], r["iteration_limit"], r["plot_per_epoch"],
             r.get("n0", -1), r.get("loss_precision", -1.0), r.get("param_mix", 0.0),
             r.get("n_models", -1), r.get("p_dropout", -1.0), r.get("reinit_on_grow", False))
            for r in rows}
    print(f"resume: {len(done)} cells already in {arguments.output}")

  base_precision = float(config["training"]["loss_precision"]) if arguments.precision is None else arguments.precision
  print(f"designs: {list(designs)}")
  print(f"seeds {arguments.seeds} | weight_decay {arguments.weight_decay} | "
        f"iteration_limit {arguments.iteration_limit} | precision {base_precision:g} | "
        f"n0 {arguments.n0}")

  # The three SWEEP axes as one product, so adding a third does not push the loop body a level right.
  for iteration_limit, weight_decay in itertools.product(arguments.iteration_limit,
                                                         arguments.weight_decay):
    for design_name, (x_scaled, proxy_loss) in designs.items():
      for seed in arguments.seeds:
        for plot in arguments.plot_per_epoch:
          run = run_config_for(config, weight_decay=weight_decay, iteration_limit=iteration_limit,
                               precision=arguments.precision, device=arguments.device,
                               n0=arguments.n0, n_increment=arguments.n_increment,
                               warmup_epochs=arguments.warmup_epochs, patience=arguments.patience,
                               n_models=arguments.n_models, p_dropout=arguments.p_dropout,
                               reinit_on_grow=arguments.reinit_on_grow, param_mix=arguments.param_mix,
                               plot_per_epoch=plot)
          if arguments.prior_scale is not None:
            run["training"]["prior_scale"] = float(arguments.prior_scale)
          if arguments.param_mix is not None:
            run["training"]["param_mix"] = float(arguments.param_mix)
          (optimizer_name, ), = (run["training"]["optimizer"].keys(), )
          (regressor_name, ), = (run["regressor"].keys(), )
          # Either may be ABSENT from a config, in which case the regressor's own defaults apply (a
          # single net, no dropout); -1 records "not declared", distinctly from every real value.
          regressor = run["regressor"][regressor_name]
          n_models_set = int(regressor.get("n_models", -1))
          p_dropout_set = float(regressor.get("p_dropout", -1.0))
          cell = (design_name, int(seed), float(run["training"]["optimizer"][optimizer_name]["weight_decay"]),
                  int(run["training"]["iteration_limit"]), int(plot),
                  int(run["training"]["n0"]), float(run["training"]["loss_precision"]),
                  float(run["training"].get("param_mix", 0.0)), n_models_set, p_dropout_set,
                  bool(run["training"].get("reinit_on_grow", False)))
          if cell in done:
            print(f"[skip] {cell}", flush=True)
            continue
          print(f"\n[cell] design={design_name} seed={seed} weight_decay={cell[2]:g} "
                f"iteration_limit={cell[3]} plot_per_epoch={plot} n0={cell[5]} precision={cell[6]:g} "
                f"n_models={cell[8]} p_dropout={cell[9]:g} reinit_on_grow={cell[10]} "
                f"param_mix={cell[7]:g}", flush=True)
          plot_dir = os.path.join(arguments.plots_dir, f"{design_name}-s{seed}") if plot == 1 else None
          row = measure(detector, run, x_scaled, int(seed), plot_dir=plot_dir)
          row.update({"prior_scale": float(run["training"].get("prior_scale", 0.0)),
                      "param_mix": float(run["training"].get("param_mix", 0.0)),
                      "design": design_name, "seed": int(seed), "weight_decay": cell[2],
                      "iteration_limit": cell[3], "plot_per_epoch": int(plot),
                      "loss_precision": float(run["training"]["loss_precision"]),
                      "n_increment": int(run["training"]["n_increment"]),
                      "warmup_epochs": int(run["training"]["warmup_epochs"]),
                      "patience": int(run["training"]["patience"]),
                      # Recorded even when unset, so every row states which RULE produced it rather
                      # than leaving it to be inferred from the submission time.
                      # Same reason for the network knobs: an arm is identified by its ROW, never by
                      # which command someone remembers having submitted.
                      "reinit_on_grow": bool(run["training"].get("reinit_on_grow", False)),

                      "n_models": n_models_set, "p_dropout": p_dropout_set,
                      "x_scaled": np.asarray(x_scaled, np.float64).tolist(),
                      "proxy_loss": proxy_loss})
          rows.append(row)
          done.add(cell)
          print(f"  -> {row['status']}: diff={row.get('diff', float('nan')):.5f} "
                f"err={row.get('err', float('nan')):.5f} slack={row.get('slack', float('nan')):.5f} "
                f"| train={row.get('train', float('nan')):.4f} val={row.get('val', float('nan')):.4f} "
                f"| window={row.get('window', -1)} spent={row['spent']} {row['wall_s']:.1f}s", flush=True)
          os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
          with open(arguments.output, "w") as f:  # after EVERY cell: a killed job keeps its work
            json.dump({"config": name, "landscape": arguments.landscape,
                       "overrides": list(arguments.overrides), "rows": rows}, f, indent=2, default=float)

  measured = [r for r in rows if r["status"] in ("converged", "capped") and "slack" in r]
  print(f"\nwrote {arguments.output} ({len(rows)} cells)")
  if len(measured) > 0:
    worst = max(measured, key=lambda r: r["slack"])
    capped = [r["design"] for r in measured if r["status"] == "capped"]
    print(f"MAXIMUM slack over cells: {worst['slack']:.5f} at design={worst['design']} seed={worst['seed']} "
          f"weight_decay={worst['weight_decay']:g} iteration_limit={worst['iteration_limit']} "
          f"({worst['status']})")
    print(f"hit the cap: {sorted(set(capped))}")
  print("READ IT AS: a CONVERGED cell is CENSORED at its `loss_precision` -- it proves the design can "
        "reach that value, not what its floor is. Only a CAPPED cell reports an uncensored slack, and "
        "the smallest precision the task can be run at is above the largest of those.")


if __name__ == "__main__":
  main()
