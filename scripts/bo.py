#!/usr/bin/env python3
"""Bayesian optimisation of the SST detector design.

Thin driver around two reusable wrappers:

* :class:`detopt.bo.BayesianOptimizer` -- the outer GP+EI loop. BO searches the
  *scaled* cube ``[0, 1]^d``, as the subgradient and LFI methods do: each coordinate
  is its own design range affinely, so the GP is fitted and EI maximised directly on
  what the caller passes and a uniform draw IS a uniform design. NOMINAL (physical)
  designs are loaded from the config and written to the trajectory file, but are never
  searched.
* :class:`detopt.nn.trainer.DesignTrainer` -- per-design network training. It
  builds JIT train/eval kernels once, preallocates fixed-size GPU event pools,
  runs the data-growing convergence procedure (see its docstring), and
  checkpoints once per design, at convergence. The detector-call budget is
  shared across all designs.

The objective is the converged VALIDATION loss (BO minimises it directly); each
observation's GP noise is that estimate's own uncertainty, ``|val - train| +
hypot(sems)``.

THE DESIGN PENALTY, when a detector prices one. ``detector.design_penalty(design)`` returns a scalar
added to the trained loss, or ``None`` when the task has no such term -- and ``None`` is DROPPED
rather than coerced to ``0.0``, so a task without a price never reports one. The term is
deterministic in the design, so it moves the mean the GP sees and leaves ``noise`` alone. It is
applied HERE and never inside the trainer: ``loss_precision`` and the settled test judge the
NETWORK's fit, and a per-design constant added there would shift the reported loss without changing
anything the criterion measures. ``results.json`` carries ``trained_loss`` and ``design_penalty``
beside ``loss`` so the two stay separable after the fact.

THE OBSERVATION-NOISE SCALE. ``bo.observation_noise_scale`` (default 1.0) multiplies the noise every
observation is handed to the GP. The trainer reports the loss estimate's OWN standard error, which
prices the estimator and not the run-to-run scatter of retraining the same design, so the GP can be
told a noise well below the scatter it actually sees -- and a GP under-told its noise interpolates
its observations instead of smoothing them. The knob exists to MEASURE that. At the default the
trainer's value is passed through unchanged, so the default path is the one that was always run.

SEEDS AND RESUME
----------------
One :class:`numpy.random.SeedSequence` per run, split ONCE before any iteration into a network branch
and an iteration branch. The network branch seeds the trainer's construction (the regressor's own rng
and the run's train/val event split); the iteration branch yields ONE seed per iteration, which seeds
BOTH that iteration's proposal and its training. Nothing holds a generator across iterations, so a
run is a function of (root seed, iteration index) and resuming is replaying the branch k times.

An interrupted run therefore restarts at the DESIGN BOUNDARY: the design it died on is proposed and
trained again from scratch, and everything that crossed a boundary -- the optimiser's evidence and the
event pools -- comes back two different ways. The optimiser's evidence is read from
``optimizer.npz``, written as one generation (stage, rename the previous generation aside, rename the
new one in, delete the old). The pools are NOT stored: ``Trainer.replay`` re-simulates the committed
trajectory, which reproduces them exactly because ``detector(design, event_index)`` is deterministic
and the event index is a function of (detector size, seed, generations). A pool can therefore never
disagree with the optimiser -- it is derived from the very rows the optimiser paid for -- and a
budget-sized state file no longer has to be carried beside every run.
"""

import json
import os
import sys
import time

import matplotlib
import yaml

matplotlib.use("AGG")  # before any pyplot import (detopt.utils.viz pulls it in)

import numpy as np

import detopt
import detopt.bo
import detopt.utils.io
from detopt.bo import BayesianOptimizer
from detopt.nn.trainer import (
  ContinualRandomFrozenTrainer, ContinualRandomOnlineTrainer, ContinualRatioTrainer, ContinualReinitTrainer, ContinualTrainer,
  DesignTrainer
)
from detopt.utils.viz.bo import plot_iteration, plot_convergence

# from_scratch / continue / closest all use DesignTrainer (a fresh per-design
# network; "continue" and "closest" warm-start its weights from a previous design).
# "meta" is the ContinualTrainer: one persistent network trained with current+history
# replay. "meta_ratio" is the same continual strategy with the current:replay batch
# composition as a knob (`training.current_replay_ratio`), of which "meta"'s 50/50
# batch is the 1:1 case.
#
# WHICH ARMS SEE THE DESIGN, and why it is not the same for all of them. Each trainer answers
# `Trainer.reveals_design()` and the training procedure combines accordingly. The two continual arms
# return True: ONE network spans many designs and every batch mixes the current design with replay
# from earlier ones, so the design is the only thing telling those rows apart. The three per-design
# arms -- from_scratch / continue / closest, all `DesignTrainer` -- return False: each trains for ONE
# design at a time, so the design is CONSTANT across the whole batch and carries nothing the network
# could use. Their features are the design-free ones (`combined_event_shape(design=False)`), and the
# regressor is built for that width.
#
# THE EVENTS ARE SIMULATED AT THE TRUE DESIGN IN EVERY ARM. Withholding changes what the network is
# TOLD, never what the detector measured -- a task whose measurement depends on its design (the
# visible window, the sampling blur) still applies it, and drops only what would announce which
# design produced the reading.
VALID_INIT_STRATEGIES = (
  "from_scratch", "continue", "closest", "meta", "meta_reinit", "meta_ratio", "meta_random_frozen", "meta_random_online"
)

# THE TRAINER CLASSES THIS DRIVER USES, KEYED BY STRATEGY. "per_design" backs from_scratch /
# continue / closest (they differ only in the warm start this driver passes); every other strategy
# names its own class here, so adding one is a line in this table rather than a branch below.
TRAINERS = {
  "per_design": DesignTrainer,
  "meta": ContinualTrainer,
  "meta_reinit": ContinualReinitTrainer,
  "meta_ratio": ContinualRatioTrainer,
  "meta_random_frozen": ContinualRandomFrozenTrainer,
  "meta_random_online": ContinualRandomOnlineTrainer
}

# TRAINING KNOBS ONLY SOME TRAINERS ACCEPT, and which strategies own them. A run config written for a
# multi-arm campaign has to carry every arm's knobs, but a trainer that does not take one raises
# `TypeError: __init__() got an unexpected keyword argument` before the first design -- so a knob is
# dropped for the arms that do not own it. Dropped LOUDLY: a setting that vanishes without a line in
# the log is how a campaign ends up measuring something else.
RANDOM_ARMS = ("meta_random_frozen", "meta_random_online")
STRATEGY_KNOBS = {
  "replay_weight": ("meta", "meta_reinit", "meta_ratio") + RANDOM_ARMS,
  "current_replay_ratio": ("meta_ratio", ),
  "alpha": RANDOM_ARMS,
  "random_weight": RANDOM_ARMS
}


def drop_foreign_knobs(config, nn_init_strategy):
  """``config`` with the ``training`` knobs this arm's trainer does not accept removed.

    A run config written for a multi-arm campaign carries EVERY arm's knobs, and a trainer handed one
    it does not take dies with `TypeError: __init__() got an unexpected keyword argument` before the
    first design. Dropped LOUDLY, one line per drop: a setting that vanishes without a line in the log
    is how a run ends up measuring something else.

    Used by every entry point that builds a trainer from a campaign config -- the driver and the
    retention probe -- so the two cannot disagree about which knob belongs to which arm.
  """
  for knob, owners in STRATEGY_KNOBS.items():
    if nn_init_strategy not in owners and knob in config.get("training", {}):
      config = {**config, "training": {k: v for k, v in config["training"].items() if k != knob}}
      print(
        f"[config] dropped `training.{knob}`: it applies to {'/'.join(owners)} only, "
        f"not `{nn_init_strategy}`", flush=True
      )
  return config


# THE THREE KNOBS THAT DEFINE PARAMETER RETENTION, and the only keys a strategy config may set
# besides the arm itself. `rewind` mixes the network back toward its own initialisation at each data
# addition; `shrink`/`param_noise` are shrink-and-perturb. They are mutually exclusive and the
# trainer enforces that.
RETENTION_KNOBS = ("rewind", "shrink", "param_noise")


def _resolve_strategy(config):
  """Fold a named strategy config into the run config.

    `strategy=<name>` resolves, through gearup, to `config/strategy/<name>.yaml`: a top-level key
    whose value is a string is loaded from the directory of the same name. That file names the ARM
    and its retention knobs and nothing else, so a campaign cell is addressed as
    `=ship_angle_final strategy=angle-meta-rewind-025` instead of repeating four overrides, and the
    file that defines the arm is named in the submit line and recorded in `config.yaml`.

    THERE IS NO PRECEDENCE RULE, BY DESIGN. A task config that also sets `nn_init_strategy` or a
    retention knob is an error rather than a loser, because a campaign in which one of these is
    silently overridden measures something other than what its name says.
  """
  strategy = config.get("strategy")
  if strategy is None:
    return config

  allowed = ("nn_init_strategy", ) + RETENTION_KNOBS
  unknown = sorted(set(strategy) - set(allowed))
  if len(unknown) > 0:
    raise ValueError(f"strategy config may only set {allowed}, got extra keys {unknown}")
  if "nn_init_strategy" not in strategy:
    raise ValueError("strategy config must name `nn_init_strategy`")

  training = dict(config.get("training", {}))
  clashes = sorted(k for k in RETENTION_KNOBS if k in training)
  if len(clashes) > 0:
    raise ValueError(
      f"training sets {clashes} but a strategy config is in use; retention knobs belong to the "
      f"strategy alone -- remove them from the task config"
    )
  if config.get("nn_init_strategy") is not None:
    raise ValueError("task config sets `nn_init_strategy` but a strategy config is in use; remove it")

  for knob in RETENTION_KNOBS:
    if knob in strategy:
      training[knob] = strategy[knob]
  resolved = {**config, "nn_init_strategy": strategy["nn_init_strategy"], "training": training}
  print(
    f"[strategy] {strategy['nn_init_strategy']}; "
    + ", ".join(f"{k}={training[k]}" for k in RETENTION_KNOBS if k in training), flush=True
  )
  return resolved


def bo(output, seed: int, force: bool = False, **config):
  seed = int(seed)
  config = _resolve_strategy(config)

  # THE CONFIG THAT ACTUALLY RAN, written BEFORE anything else happens, so it exists even for a run
  # that dies in setup. A run is launched as `=<name>` plus command-line overrides, so neither the
  # config file nor the command line alone says what a finished run used. This is every argument the
  # driver received -- `output`, `seed`, `force` and the whole composed config -- and it is rewritten
  # on a resume, so it always describes the run now executing.
  os.makedirs(output, exist_ok=True)
  with open(os.path.join(output, "config.yaml"), "w") as handle:
    yaml.safe_dump({
      "output": output,
      "seed": seed,
      "force": force,
      **config
    }, handle, sort_keys=False, default_flow_style=False)

  nn_init_strategy = config.get("nn_init_strategy", "from_scratch")
  if nn_init_strategy not in VALID_INIT_STRATEGIES:
    raise ValueError(f"nn_init_strategy {nn_init_strategy!r} not in {VALID_INIT_STRATEGIES}")

  # ONE TRAJECTORY FILE, AND COMPLETION IS A MARKER BESIDE IT. `results.json` is rewritten after
  # every iteration and always holds everything measured so far; the workflow touches `done.txt`
  # once the run's budget pool has filled, and `done.txt` -- not `results.json` -- is the rule's
  # declared output. Two consequences, and both are the reason for the change:
  #
  #   * NOTHING DELETES THE TRAJECTORY. Snakemake removes a rule's declared outputs before running
  #     it, so while `results.json` was the output a rescheduled run could have the very file its
  #     resume needs cleared out from under it. Now only `done.txt` is at risk, and losing that
  #     costs a marker.
  #   * INVALIDATING A FINISHED RUN IS `rm done.txt`. The run reopens at the design it stopped on
  #     and keeps going -- which is exactly what continuing a campaign under a RAISED budget needs.
  #     Under the old scheme the same act meant either starting over or hand-editing file names.
  #
  # A run whose budget is unchanged reopens, finds its pool already full, and finishes immediately
  # having added nothing, so re-running is idempotent rather than destructive.
  #
  # `partial.json` is the OLD name for the in-flight trajectory. It is still READ, so a run
  # interrupted under the previous scheme resumes rather than being lost, and it is removed once
  # `results.json` has been written in its place.
  results_path = os.path.join(output, "results.json")
  partial_path = os.path.join(output, "partial.json")
  budget_configured = int(config["training"]["budget"])
  resume = None
  prior_path = None
  if force:
    # A forced run starts over: leave nothing for the resume path to pick up.
    for path in (results_path, partial_path):
      if os.path.exists(path):
        os.remove(path)
  elif os.path.exists(results_path):
    prior_path = results_path
  elif os.path.exists(partial_path):
    prior_path = partial_path

  if prior_path is not None:
    with open(prior_path) as f:
      prior = json.load(f)
    used = int(prior.get("detector_calls_used", 0))
    # Only the COMPLETE rows resume: the row a run stopped on records a design that was never
    # scored, and it is re-proposed rather than re-read.
    resume = detopt.utils.io.complete_results(prior.get("results", []))
    state = "complete" if prior.get("completed") is True else "in flight"
    print(
      f"[resume] {prior_path}: {len(resume)} scored designs, {used}/{budget_configured} detector "
      f"calls, previously {state}."
    )
    if detopt.utils.io.restore_path(os.path.join(output, "optimizer.npz")) is None:
      print(f"[warning] no committed state beside it -- restarting from scratch.")
      resume = None

  os.makedirs(output, exist_ok=True)
  plots_dir = os.path.join(output, "plots")
  os.makedirs(plots_dir, exist_ok=True)

  # PER-DESIGN CONVERGENCE PLOTS, as a STRIDE rather than a switch. `plot_per_epoch: 8` renders
  # every 8th epoch, `1` every epoch, `0`/`false` never. `true` means 1, so old configs are
  # unchanged. A stride exists because the render is the expensive half: `plot_iteration` draws a
  # whole figure and rewrites the design JSON on a background worker, measured at ~36% of a
  # design's wall clock when done every epoch -- host-side work performed AFTER the network has
  # converged, which on a shared box comes straight off the other jobs. At a stride of 8 the
  # diagnostic costs about an eighth of that and still shows the shape of every design's curve.
  plot_every = config.get("plot_per_epoch", 8)
  plot_every = (1 if plot_every else 0) if isinstance(plot_every, bool) else int(plot_every)
  bo_cfg = config["bo"]
  gp_cfg = dict(bo_cfg["gp"])
  ei_cfg = dict(bo_cfg["ei"])
  # Initial random proposals: as many as the GP's CV folds, so the first GP
  # fit has enough points for k-fold cross-validation.
  n_init = int(bo_cfg.get("n_init", gp_cfg["n_folds"]))
  # See "THE OBSERVATION-NOISE SCALE" in the module docstring. Announced only when it is off the
  # default, so a run that uses one cannot be mistaken for a run that does not.
  noise_scale = float(bo_cfg.get("observation_noise_scale", 1.0))
  if noise_scale != 1.0:
    print(
      f"[bo] observation_noise_scale={noise_scale!r}: the GP is told the trainer's reported "
      f"standard error times {noise_scale!r}", flush=True
    )

  detector = detopt.detector.from_config(config["detector"])
  # BO searches the SCALED design cube [0, 1]^d -- the same space the subgradient and LFI methods
  # optimise in, so the drivers stay interchangeable. Each coordinate is its own design range
  # affinely, so a uniform draw in the cube IS a uniform DESIGN and there are no bounds to pass
  # beyond the dimension. Nominal (physical) designs are loaded from the config and written to
  # results.json; they are never the search space.
  d = int(detector.design_dim())
  # The GP prior comes from `bo.gp.kernel` (detopt.bo.__kernels__). It is built HERE because it
  # needs the detector -- the design dimension, and which of its coordinates interchange -- which
  # BayesianOptimizer never sees.
  kernel = detopt.bo.kernel_from_config(gp_cfg.pop("kernel"), detector, gp_cfg)
  bo_opt = BayesianOptimizer(d, gp=gp_cfg, ei=ei_cfg, kernel=kernel, n_init=n_init)

  # ONE sequence for the run, split ONCE before any iteration: the network branch seeds the
  # regressors' own rng, which lives inside the model and is saved with it; the iteration branch
  # yields one seed per iteration, which seeds BOTH the proposal and that iteration's training.
  # Nothing else holds random state, so resuming is replaying this sequence k times -- there is no
  # generator position to persist and no way for a resumed run to drift onto a different stream.
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)

  trainer_cls = TRAINERS.get(nn_init_strategy, TRAINERS["per_design"])
  # The resolved reveal mode is NOT recoverable from the log otherwise, and a setting that decides
  # what the network is shown must never be silent -- an earlier run recorded a knob that no longer
  # exists and could not be reproduced.

  config = drop_foreign_knobs(config, nn_init_strategy)

  trainer = trainer_cls.from_config(
    detector, config, checkpoint_dir=os.path.join(output, "checkpoints"), seed=int(network_seq.generate_state(1)[0]),
  )

  reveal = trainer.reveal()
  shape = detector.combined_event_shape(reveal != "none")
  source = "training.reveal" if config["training"].get("reveal") is not None else "strategy default"
  print(f"[reveal] {nn_init_strategy} -> {reveal!r} ({source}); features {shape}", flush=True)

  # The trainer owns the budget-sized event pools (train + val); the run ends
  # when they fill. Designs append into them (windowed) across iterations.
  budget = trainer.train_pool.capacity + trainer.val_pool.capacity

  optimizer_state_path = os.path.join(output, "optimizer.npz")

  # One entry per COMPLETED iteration: the design that iteration proposed, in the scaled cube. It is
  # the warm-start index -- `closest` measures distances in it and the row it picks IS the design
  # number, so the network for that row is read from `checkpoints/design_<row>`. Historical networks
  # are never held in memory: the checkpoint is the one copy, and it is the copy that survives an
  # interruption, so this list is restored on resume and warm starts keep working across one.
  proposed_scaled = []
  results = []
  # Detector calls already charged to a row. Resume rebuilds it from the restored rows, so a resumed
  # run does not re-attribute a constructor purchase that an earlier attempt already recorded.
  attributed = 0
  best_loss, best_design = np.inf, None

  def _commit_state():
    """Publish the optimiser state.

        The event pools are NOT written: a resumed run rebuilds them with `Trainer.replay` from the
        committed trajectory, so the only state on disk is the optimiser's own evidence. The pool can
        no longer disagree with the optimiser because it is DERIVED from the rows the optimiser paid
        for.

        THE STATE PAIR IS THE AUTHORITY, and it is committed AFTER ``partial.json`` is written. A
        crash in the window between the two leaves a partial row whose events never reached the
        committed pool; the resume path drops that row and re-measures the design, which costs one
        design. The other order is not recoverable at all -- it would leave the optimiser holding an
        observation whose row does not exist, and nothing on disk can reconstruct it.
        """
    bo_opt.persist(optimizer_state_path)
    detopt.utils.io.commit([optimizer_state_path])

  start_iteration = 0
  if resume is not None:
    bo_opt.restore(optimizer_state_path)
    start_iteration = int(bo_opt.X.shape[0])
    if len(resume) < start_iteration:
      raise ValueError(
        f"{output}: state holds {start_iteration} observations but partial.json records "
        f"only {len(resume)} complete rows -- the trajectory is written FIRST, so it can "
        f"never legitimately lag the state; this pair was not produced by one run"
      )
    if len(resume) > start_iteration:
      print(
        f"[resume] dropping {len(resume) - start_iteration} trajectory row(s) written after the last "
        f"committed state; those designs are re-proposed and re-measured."
      )
      resume = resume[:start_iteration]
    # THE POOLS ARE REBUILT, NOT LOADED. `detector(design, event_index)` is deterministic and the
    # event index is a function of (detector size, seed, generations), so re-simulating the
    # committed rows reproduces the pools exactly -- verified leaf for leaf, pools and carried
    # network, by `scripts/verify_replay.py`. Only the rows the OPTIMISER has paid for are
    # replayed, which is what keeps the old guarantee that a resumed run never holds an
    # observation its pool has not paid for.
    trainer.replay(resume)
    results = list(resume)
    attributed = sum(int(row.get("spent", 0)) for row in results)
    proposed_scaled = [np.asarray(r["x_scaled"], dtype=np.float32) for r in resume]
    scored = [float(r["loss"]) for r in resume]
    if len(scored) > 0:
      best_index = int(np.argmin(scored))
      best_loss, best_design = scored[best_index], resume[best_index]["design"]
    # REPLAY, do not re-derive: the iteration seed of step k is the k-th spawn of this branch, so
    # skipping k spawns puts the resumed run on exactly the stream it would have been on.
    iteration_seq.spawn(start_iteration)
    print(
      f"[resume] {output}: continuing at iteration {start_iteration} "
      f"({trainer.spent_calls()}/{budget} detector calls spent, "
      f"best={best_loss:.5f})"
    )

  def _save_results(n_completed, completed):
    """Dump the trajectory to ``results.json``, in flight and at the end alike.

        ONE FILE, AND COMPLETION IS A SEPARATE MARKER. `results.json` is written after every
        iteration and always holds everything measured so far; `done.txt` is touched by the workflow
        only once the run's budget pool has filled. The file name no longer carries the completion
        signal, and that is the point: `results.json` is never a workflow OUTPUT, so nothing deletes
        it when a rule is rescheduled, and a resumed run always has the full trajectory to read back.
        INVALIDATING A FINISHED RUN IS THEN `rm done.txt` -- the run reopens exactly where it stopped
        instead of starting over, which is what continuing a campaign under a raised budget needs.
        The `completed` field inside the file records the same fact for readers that only have the
        JSON.

        ONLY COMPLETED ITERATIONS ARE STORED. Every entry has a real ``loss`` and ``spent``, so a
        consumer can do arithmetic on the array without filtering it first, and the FILENAME is the
        only completeness signal there is. The design a run stopped on is not recorded at all: it was
        never trained, the seed sequence re-proposes it exactly on resume, and it is in the log.

        An earlier version put that design in ``results`` with null fields. Because budget exhaustion
        is the NORMAL end of a run, every finished trajectory then carried one, and
        ``np.array([..., None], dtype=float)`` yields NaN rather than raising -- so the plots did not
        crash, they ended in a NaN that ``cumsum`` smeared backwards, and ``verify_trajectory`` died on
        ``int(None)``. ``detopt.utils.io.complete_results`` remains, and readers still go through it,
        only because files written under that scheme are already on disk.
        """
    payload = {
      "results": results,
      "best_loss": float(best_loss),
      "best_design": best_design,
      "n_iterations_completed": n_completed,
      "detector_calls_used": int(trainer.spent_calls()),
      "method": "JAX-GP+EI",
      "completed": completed,
      # THE CONFIG THIS TRAJECTORY WAS PRODUCED UNDER, recorded because without it a
      # `results.json` is not self-describing and cross-campaign comparisons cannot be checked
      # from the artefacts. MEASURED COST of its absence, 2026-08-20: two campaigns 2.67x apart
      # in cold-start cost and 0.14 apart in loss level looked mutually comparable, and one was
      # used to overturn a correct finding; recovering the truth needed four separate forensic
      # signals (cold-start spend, loss level, a missing run.log, directory mtime) where one
      # field settles it. `run.log` is not a fallback -- an older campaign here has none at all.
      "config": config,
      "nn_init_strategy": nn_init_strategy,
    }
    staged = results_path + ".new"
    with open(staged, "w") as f:
      json.dump(payload, f, indent=2, default=float)
      f.flush()
      os.fsync(f.fileno())
    os.replace(staged, results_path)
    if os.path.exists(partial_path):
      os.remove(partial_path)

  print(
    f"BO: running until the budget pool fills "
    f"(budget={budget} detector calls, n_init={n_init}, d={d}, plot_per_epoch={plot_every})"
  )

  i = start_iteration
  while True:
    iter_start = time.time()
    iteration_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])
    x_prop = np.asarray(bo_opt.propose(iteration_seed), dtype=np.float32)
    if bo_opt.last_info is not None:
      info = bo_opt.last_info
      print(
        f"  [acq] EI={info['ei']:.4g} "
        f"| log_ls~{info['log_lengthscale_mean']:.3f} "
        f"log_amp={info['log_amplitude']:.3f}"
      )

    design_phys = np.asarray(detector.flatten_design(detector.to_nominal(x_prop)), dtype=np.float32).tolist()

    # The LAST snapshot a design produced, kept so the final curve can be drawn once the design
    # has exited. The stride alone renders every `plot_every`-th epoch, which leaves a short
    # design showing one or two points and its exit epoch missing entirely -- the very shape one
    # needs to see to tell "converged" from "never started", since a flat curve at the class
    # prior passes the settled test more easily than a descending one.
    last_snapshot = {}

    def _on_epoch(snapshot, _i=i, _d=design_phys, _keep=last_snapshot):
      vlp = snapshot["val_loss_per_epoch"]
      _keep["snapshot"], _keep["iteration"], _keep["design"] = snapshot, _i, _d
      # The snapshot's length IS the epoch count, so the stride gates here. The FIRST epoch and
      # every `plot_every`-th one are drawn: without the first, a design that converges inside
      # one stride would produce no plot at all. The stride gates the RENDER ONLY -- the arrays
      # are kept above it, so `plot_per_epoch: 0` still saves the curve.
      if plot_every <= 0:
        return
      epoch = int(vlp.size)
      if epoch != 1 and epoch % plot_every != 0:
        return
      live = float(vlp[-1]) if vlp.size > 0 else float("nan")
      plot_iteration(snapshot, iteration=_i, design=_d, val_loss=live, plots_dir=plots_dir)

    def _plot_final(_keep=last_snapshot):
      """Persist the design's convergence CURVE, and render its LAST epoch.

            The npz is written unconditionally and is the primary artefact: a PNG cannot be
            re-analysed, and without the arrays a finished run cannot answer afterwards at what
            window it converged or whether a width was capacity- or data-limited. It carries every
            per-epoch series the trainer reported, including `train_budget_per_epoch`, which is
            where the data injections are.
            """
      snapshot = _keep.get("snapshot")
      if snapshot is None:
        return
      os.makedirs(plots_dir, exist_ok=True)
      np.savez_compressed(
        os.path.join(plots_dir, f"iter_{_keep['iteration']:03d}_history.npz"), **{
          k: np.asarray(v)
          for k, v in snapshot.items() if not isinstance(v, (str, bytes))
        },
      )
      vlp = snapshot["val_loss_per_epoch"]
      live = float(vlp[-1]) if vlp.size > 0 else float("nan")
      if plot_every > 0:
        plot_iteration(snapshot, iteration=_keep["iteration"], design=_keep["design"], val_loss=live, plots_dir=plots_dir)

    on_epoch = _on_epoch

    # Network init strategy (todo.md): from_scratch trains fresh; continue
    # warm-starts from the previous design; closest from the nearest previously trained design
    # by L2 in the SCALED cube -- a change of metric from the old encoded L2: distances are now
    # uniform across each range instead of stretched near the bounds.
    # Warm-start applies only to the per-design DesignTrainer strategies; the
    # "meta" ContinualTrainer carries its own persistent network.
    # THE NETWORK IS READ FROM THAT DESIGN'S CHECKPOINT, never from a list kept here. The driver
    # holds no historical parameters at all: the checkpoint written at convergence is the one copy
    # and it is on disk, so the warm-start pool is whatever the run has MEASURED rather than
    # whatever this process happens to remember -- and a resumed run warm-starts from designs
    # scored before the interruption exactly as an uninterrupted one does.
    init_params = None
    warm_from = None
    if len(proposed_scaled) > 0 and nn_init_strategy in ("continue", "closest"):
      if nn_init_strategy == "continue":
        warm_from = len(proposed_scaled) - 1
      else:  # closest
        dists = np.linalg.norm(np.asarray(proposed_scaled) - x_prop[None, :], axis=1)
        warm_from = int(np.argmin(dists))
        print(f"  [warm-start] closest = iter {warm_from} (dist={float(dists[warm_from]):.3f})")
      init_params = trainer.restore_design_parameters(warm_from)

    used = trainer.spent_calls()
    print(f"[iter {i+1}] training... ({budget - used} detector calls left)")
    # THE SPEND THIS ITERATION IS THE TRAINER'S, NOT THE DESIGN'S. `TrainResult.spent` counts what the
    # proposed design consumed; a strategy may buy other events out of the same budget, and those are
    # detector calls the run has to answer for. The delta is measured against everything ALREADY
    # ATTRIBUTED to a row rather than against the fill at the top of this iteration, because a strategy
    # may also buy events in its CONSTRUCTOR -- `meta_random_frozen` buys its whole random prefix before
    # iteration 1 exists, and a per-iteration bracket misses it entirely, leaving `sum(spent)` short of
    # `detector_calls_used` by exactly `alpha * budget` and the arm's curve wrongly shifted left.
    # Anything unattributed lands on the FIRST row, which is where the spend actually happened.
    # Arms that buy nothing extra see a delta of exactly `result.spent`, so their records are unchanged.
    try:
      result = trainer.train(x_prop, iteration_seed, init_params=init_params, on_epoch=on_epoch, step=i, )
    except RuntimeError as error:
      if plot_every > 0:
        _plot_final()
      text = str(error).replace("\n", " ")
      if "did not reach precision within iteration_limit" not in text:
        raise
      _save_results(i, False)
      print(
        f"[failed] design {i} did not reach precision after {i} scored designs; "
        f"partial.json holds those, and this design is re-proposed on resume"
      )
      raise
    if plot_every > 0:
      _plot_final()
    if result is None:
      print(f"[budget] pool exhausted; finishing BO after {i} completed iterations.")
      break
    # Detector calls this iteration bought BESIDES the proposed design's own window.
    spent_random = max(0, (trainer.spent_calls() - attributed) - int(result.spent))
    trained_loss = float(result.objective_loss)
    penalty = detector.design_penalty(design_phys)
    penalty = None if penalty is None else float(penalty)
    loss = trained_loss if penalty is None else trained_loss + penalty

    # At the default scale the trainer's own value is passed through -- not a product with 1.0,
    # which would change its type and is not what earlier campaigns ran.
    noise = result.objective_std if noise_scale == 1.0 else result.objective_std * noise_scale
    bo_opt.append(x_prop, loss, noise=noise)
    proposed_scaled.append(x_prop)

    improved = loss < best_loss
    if improved:
      best_loss, best_design = loss, design_phys

    elapsed = time.time() - iter_start
    marker = " BEST" if improved else ""
    priced = "" if penalty is None else f" (trained {trained_loss:.5f} + penalty {penalty:.5f})"
    print(
      f"[iter {i+1}] loss={loss:.5f}±{result.objective_std:.4f}{priced} "
      f"spent={result.spent}{f' +{spent_random} random' if spent_random > 0 else ''} "
      f"time={elapsed:.1f}s{marker}"
    )

    results.append({
      "iteration": i,
      "design": design_phys,
      "x_scaled": x_prop.tolist(),
      "loss": loss,
      "trained_loss": trained_loss,
      "design_penalty": penalty,
      "loss_std": float(result.objective_std),
      "spent": int(result.spent) + spent_random,
      "spent_train": int(result.spent_train),
      "spent_val": int(result.spent_val),
      "spent_random": spent_random,
      "time_s": float(elapsed),
      "nn_init_strategy": nn_init_strategy,
      "warm_start_from": warm_from,
    })
    attributed += int(result.spent) + spent_random

    # THE RECORD FIRST, then the state -- see `_commit_state` for why this order is the
    # recoverable one.
    _save_results(i + 1, completed=False)
    _commit_state()

    # Refresh the convergence plot after every completed iteration.
    plot_convergence(results, output)
    i += 1

  _save_results(i, completed=True)  # the budget pool filled -- reruns skip this output
  plot_convergence(results, output)
  print(f"\nBest loss: {best_loss:.6f}")
  return best_loss, best_design, results


if __name__ == "__main__":
  import sys

  import gearup

  # gearup's CLI is `key=value`; `--force` is the conventional spelling, translated here.
  arguments = ["force=yes" if a == "--force" else a for a in sys.argv[1:]]
  gearup.gearup(bo).with_config("config/root.yaml")(arguments)
