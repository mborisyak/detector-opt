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

SEEDS AND RESUME
----------------
One :class:`numpy.random.SeedSequence` per run, split ONCE before any iteration into a network branch
and an iteration branch. The network branch seeds the trainer's construction (the regressor's own rng
and the run's train/val event split); the iteration branch yields ONE seed per iteration, which seeds
BOTH that iteration's proposal and its training. Nothing holds a generator across iterations, so a
run is a function of (root seed, iteration index) and resuming is replaying the branch k times.

An interrupted run therefore restarts at the DESIGN BOUNDARY: the design it died on is proposed and
trained again from scratch, and everything that crossed a boundary -- the optimiser's evidence and the
event pools -- is read back from ``optimizer.npz`` / ``trainer.npz``. Those two are written as one
generation (stage every file, rename the previous generation aside, rename the new one in, delete the
old), so no crash can resume the optimiser against a pool that has not paid for its observations.
"""

import json
import os
import sys
import time

import matplotlib

matplotlib.use("AGG")  # before any pyplot import (detopt.utils.viz pulls it in)

import numpy as np

import detopt
import detopt.bo
import detopt.utils.io
from detopt.bo import BayesianOptimizer
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils.viz.bo import plot_iteration, plot_convergence

# from_scratch / continue / closest all use DesignTrainer (a fresh per-design
# network; "continue" and "closest" warm-start its weights from a previous design).
# "meta" is the ContinualTrainer: one persistent network trained with current+history
# replay. All are design-conditioned (combine sees each event's real scaled design).
VALID_INIT_STRATEGIES = ("from_scratch", "continue", "closest", "meta")


def bo(output, seed: int, force: bool = False, **config):
    seed = int(seed)
    nn_init_strategy = config.get("nn_init_strategy", "from_scratch")
    if nn_init_strategy not in VALID_INIT_STRATEGIES:
        raise ValueError(f"nn_init_strategy {nn_init_strategy!r} not in {VALID_INIT_STRATEGIES}")

    # TWO FILES, AND `results.json` MEANS FINISHED. A run in progress writes `partial.json`; only a
    # run whose budget pool filled writes `results.json`, and `partial.json` is removed at that
    # moment. So the mere EXISTENCE of `results.json` is the completion flag, and no consumer -- or
    # build system -- ever sees a half-written trajectory under the name it asks for. Snakemake in
    # particular declares `results.json` as this rule's output: a killed run leaves that output
    # absent, so the rule is simply rescheduled and `bo.py` resumes from `partial.json`, instead of
    # the run being flagged incomplete and cleared by DELETING the very file the resume needs.
    #
    # The `completed` field inside the file is kept for readers written against the old single-file
    # layout, where >= 90% of the budget spent was the only way to infer completion.
    results_path = os.path.join(output, "results.json")
    partial_path = os.path.join(output, "partial.json")
    budget_configured = int(config["training"]["budget"])
    resume = None
    if force:
        # A forced run starts over: leave no partial for the resume path to pick up.
        if os.path.exists(partial_path):
            os.remove(partial_path)
    elif os.path.exists(results_path) and json.load(open(results_path)).get("completed") is False:
        # AN OLD-SCHEME PARTIAL. Before the split, an unfinished run wrote `results.json` with
        # `completed: false`; now the FILENAME carries that, so such a file would be mistaken for a
        # finished run and silently skipped. Treat it as the partial it is -- which also means a run
        # interrupted under the old code resumes rather than being lost.
        with open(results_path) as f:
            prior = json.load(f)
        used = int(prior.get("detector_calls_used", 0))
        print(f"[migrate] {results_path}: written by the pre-`partial.json` code and NOT complete "
              f"({used}/{budget_configured} calls) -- reading it as a partial.")
        resume = detopt.utils.io.complete_results(prior.get("results", []))
        if detopt.utils.io.restore_path(os.path.join(output, "optimizer.npz")) is None:
            print(f"[warning] no state beside it -- restarting from scratch.")
            resume = None
    elif os.path.exists(results_path):
        with open(results_path) as f:
            prior = json.load(f)
        used = int(prior.get("detector_calls_used", 0))
        print(
            f"[skip] {results_path}: run already completed "
            f"({prior.get('n_iterations_completed', 0)} iterations, best={prior.get('best_loss', float('nan')):.5f}, "
            f"{used}/{budget_configured} detector calls). Pass --force (or delete results.json) to re-run."
        )
        return prior.get("best_loss"), prior.get("best_design"), prior.get("results")
    elif os.path.exists(partial_path):
        with open(partial_path) as f:
            prior = json.load(f)
        used = int(prior.get("detector_calls_used", 0))
        # Only the COMPLETE rows resume: the row the run stopped on records a design that was never
        # scored, and it is re-proposed rather than re-read.
        resume = detopt.utils.io.complete_results(prior.get("results", []))
        if detopt.utils.io.restore_path(os.path.join(output, "optimizer.npz")) is None:
            print(f"[warning] {partial_path}: previous run is INCOMPLETE ({used}/{budget_configured} detector "
                  f"calls) and left no state to resume from -- restarting from scratch.")
            resume = None

    os.makedirs(output, exist_ok=True)
    plots_dir = os.path.join(output, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # PER-EPOCH plotting is a diagnostic and it is NOT free: `plot_iteration` renders a whole figure
    # (and rewrites the design JSON) once an EPOCH on a background worker, and the design blocks on
    # that queue when it exits. Measured on one design of this task it was ~36% of the wall clock,
    # host-side CPU work done AFTER the network had converged -- which on a shared node is taken
    # directly off the CPU-bound screening jobs. It is therefore a SETTING (`plot_per_epoch`), left
    # ON by default so no existing run config changes behaviour, and turned off in the configs of
    # runs that do not want it. Off means no callback at all, so the trainer also skips building the
    # per-epoch history snapshot. The END-OF-DESIGN convergence plot is unaffected either way.
    plot_per_epoch = bool(config.get("plot_per_epoch", True))
    bo_cfg = config["bo"]
    gp_cfg = dict(bo_cfg["gp"])
    ei_cfg = dict(bo_cfg["ei"])
    # Initial random proposals: as many as the GP's CV folds, so the first GP
    # fit has enough points for k-fold cross-validation.
    n_init = int(bo_cfg.get("n_init", gp_cfg["n_folds"]))

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

    trainer_cls = ContinualTrainer if nn_init_strategy == "meta" else DesignTrainer
    trainer = trainer_cls.from_config(
        detector,
        config,
        checkpoint_dir=os.path.join(output, "checkpoints"),
        seed=int(network_seq.generate_state(1)[0]),
    )

    # The trainer owns the budget-sized event pools (train + val); the run ends
    # when they fill. Designs append into them (windowed) across iterations.
    budget = trainer.train_pool.capacity + trainer.val_pool.capacity

    optimizer_state_path = os.path.join(output, "optimizer.npz")
    trainer_state_path = os.path.join(output, "trainer.npz")

    proposed_scaled = []  # row-aligned with trained_params (warm-start)
    trained_params = []
    results = []
    best_loss, best_design = np.inf, None

    def _commit_state():
        """Publish the optimiser and the trainer as ONE generation.

        Both are staged first and renamed into place together, so a run killed here can never come
        back with the optimiser holding an observation the event pool has not paid for.

        THE STATE PAIR IS THE AUTHORITY, and it is committed AFTER ``partial.json`` is written. A
        crash in the window between the two leaves a partial row whose events never reached the
        committed pool; the resume path drops that row and re-measures the design, which costs one
        design. The other order is not recoverable at all -- it would leave the optimiser holding an
        observation whose row does not exist, and nothing on disk can reconstruct it.
        """
        bo_opt.persist(optimizer_state_path)
        trainer.persist(trainer_state_path)
        detopt.utils.io.commit([optimizer_state_path, trainer_state_path])

    start_iteration = 0
    if resume is not None:
        bo_opt.restore(optimizer_state_path)
        trainer.restore(trainer_state_path)
        start_iteration = int(bo_opt.X.shape[0])
        if len(resume) < start_iteration:
            raise ValueError(f"{output}: state holds {start_iteration} observations but partial.json records "
                             f"only {len(resume)} complete rows -- the trajectory is written FIRST, so it can "
                             f"never legitimately lag the state; this pair was not produced by one run")
        if len(resume) > start_iteration:
            print(f"[resume] dropping {len(resume) - start_iteration} trajectory row(s) written after the last "
                  f"committed state; those designs are re-proposed and re-measured.")
            resume = resume[:start_iteration]
        results = list(resume)
        proposed_scaled = [np.asarray(r["x_scaled"], dtype=np.float32) for r in resume]
        scored = [float(r["loss"]) for r in resume]
        if len(scored) > 0:
            best_index = int(np.argmin(scored))
            best_loss, best_design = scored[best_index], resume[best_index]["design"]
        # REPLAY, do not re-derive: the iteration seed of step k is the k-th spawn of this branch, so
        # skipping k spawns puts the resumed run on exactly the stream it would have been on.
        iteration_seq.spawn(start_iteration)
        # The per-design networks themselves are NOT kept (they are a design's private result, not run
        # state), so the warm-start strategies have no history to start from on the first resumed
        # design. It trains cold; every later one warm-starts normally.
        if nn_init_strategy in ("continue", "closest"):
            print(f"[resume] {nn_init_strategy}: no stored per-design networks, so iteration "
                  f"{start_iteration} starts cold; later iterations warm-start as usual.")
        print(f"[resume] {output}: continuing at iteration {start_iteration} "
              f"({trainer.train_pool.current + trainer.val_pool.current}/{budget} detector calls spent, "
              f"best={best_loss:.5f})")

    def _save_results(n_completed, completed):
        """Dump the trajectory: to ``partial.json`` while the run is in flight, to ``results.json``
        once its budget pool has filled -- and then remove the partial, so exactly one of the two
        exists and its NAME says which.

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
        with open(results_path if completed else partial_path, "w") as f:
            json.dump(
                {
                    "results": results,
                    "best_loss": float(best_loss),
                    "best_design": best_design,
                    "n_iterations_completed": n_completed,
                    "detector_calls_used": int(trainer.train_pool.current + trainer.val_pool.current),
                    "method": "JAX-GP+EI",
                    "completed": completed,
                },
                f,
                indent=2,
                default=float,
            )
        if completed and os.path.exists(partial_path):
            os.remove(partial_path)

    print(
        f"BO: running until the budget pool fills "
        f"(budget={budget} detector calls, n_init={n_init}, d={d}, plot_per_epoch={plot_per_epoch})"
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

        def _on_epoch(snapshot, _i=i, _d=design_phys):
            vlp = snapshot["val_loss_per_epoch"]
            live = float(vlp[-1]) if vlp.size > 0 else float("nan")
            plot_iteration(snapshot, iteration=_i, design=_d, val_loss=live, plots_dir=plots_dir)

        on_epoch = _on_epoch if plot_per_epoch else None

        # Network init strategy (todo.md): from_scratch trains fresh; continue
        # warm-starts from the previous design; closest from the nearest previously trained design
        # by L2 in the SCALED cube -- a change of metric from the old encoded L2: distances are now
        # uniform across each range instead of stretched near the bounds.
        # Warm-start applies only to the per-design DesignTrainer strategies; the
        # "meta" ContinualTrainer carries its own persistent network.
        init_params = None
        warm_from = None
        if len(trained_params) > 0 and nn_init_strategy in ("continue", "closest"):
            if nn_init_strategy == "continue":
                warm_from = len(trained_params) - 1
            else:  # closest
                dists = np.linalg.norm(np.asarray(proposed_scaled) - x_prop[None, :], axis=1)
                warm_from = int(np.argmin(dists))
                print(f"  [warm-start] closest = iter {warm_from} (dist={float(dists[warm_from]):.3f})")
            init_params = trained_params[warm_from]

        used = trainer.train_pool.current + trainer.val_pool.current
        print(f"[iter {i+1}] training... ({budget - used} detector calls left)")
        try:
            result = trainer.train(
                x_prop,
                iteration_seed,
                init_params=init_params,
                on_epoch=on_epoch,
                step=i,
            )
        except RuntimeError as error:
            text = str(error).replace("\n", " ")
            if "did not reach precision within iteration_limit" not in text:
                raise
            _save_results(i, False)
            print(f"[failed] design {i} did not reach precision after {i} scored designs; "
                  f"partial.json holds those, and this design is re-proposed on resume")
            raise
        if result is None:
            print(f"[budget] pool exhausted; finishing BO after {i} completed iterations.")
            break
        loss = float(result.objective_loss)  # BO minimises the loss directly

        bo_opt.append(x_prop, loss, noise=result.objective_std)
        proposed_scaled.append(x_prop)
        trained_params.append(result.params)

        improved = loss < best_loss
        if improved:
            best_loss, best_design = loss, design_phys

        elapsed = time.time() - iter_start
        marker = " BEST" if improved else ""
        print(
            f"[iter {i+1}] loss={result.objective_loss:.5f}±{result.objective_std:.4f} "
            f"spent={result.spent} time={elapsed:.1f}s{marker}"
        )

        results.append(
            {
                "iteration": i,
                "design": design_phys,
                "x_scaled": x_prop.tolist(),
                "loss": float(result.objective_loss),
                "loss_std": float(result.objective_std),
                "spent": int(result.spent),
                "time_s": float(elapsed),
                "nn_init_strategy": nn_init_strategy,
                "warm_start_from": warm_from,
            }
        )
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
    gearup.gearup(bo).with_config("config/bo.yaml")(arguments)
