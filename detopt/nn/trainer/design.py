"""Per-design trainer with a data-growing convergence procedure.

:class:`DesignTrainer` turns a *design* into a converged objective-loss estimate:
a fresh (or warm-started) network is trained on the design's window, growing the
window until the loss estimate is precise enough.

THE CONVERGENCE PROCEDURE -- SACRED
-----------------------------------
GIVEN BY PEOPLE WHO KNOW LOGIC. It is a specification, not an implementation detail. Do not edit it,
reorder it, add a branch to it, retune a constant in it, or make any part of it configurable. There is
exactly one procedure and no way to select another.

The logic, plainly: **converged** means the training loss has converged AND the gap is small.
NOT converged means either the training loss is still decreasing, or the gap is large. A large gap --
or one expected to be large -- means add data. A still-decreasing training loss means keep training.
The ORDER is what makes it efficient: intervene early, before wasting time converging (and overfitting)
on a window whose gap is obviously too large.

After the warmup, at the end of each epoch, with ``gap = |val - train| + err`` (the spread of the two
numbers plus the error of their means -- exactly what is reported as the objective's uncertainty) and
``err = sqrt(train_sem^2 + val_sem^2)``, over the ``+patience`` horizon:

  1. ``P(gap at +patience > loss_precision) > 0.9``                    -> **add data**
  2. ``P(train change over +patience < loss_precision / 2) > 0.9``
     2.1 ``gap > loss_precision``                                      -> **add data**
     2.2 otherwise                                                     -> **return**
  3. otherwise                                                         -> **train on**

Both probabilities are posterior statements from :func:`~detopt.utils.training.bayesian_trend` -- a
conjugate linear fit to this round's post-warmup history with the measured per-epoch SEMs as KNOWN
observation noise and harmonic ``1/k`` weights -- read by :func:`probability_above` for (1) and
:func:`probability_change_below` for (2).

WHY EACH PIECE IS WHAT IT IS. Every one was paid for with a measurement, so none of it is free to
"simplify":

* **(2.1) is a COMPARISON, not a probability.** An earlier version asked ``P(gap < precision) > 0.9``
  there and HUNG on 3/5 synthetic seeds: the procedure's own dynamics drive the gap TO the precision,
  which is exactly where that probability sits at ~0.5 forever. A measured value against a threshold
  always resolves; a probability need not.
* **The ORDER is load-bearing**, and it is also what removes that band -- when (1) does not fire
  because its probability is near 0.5, (2) is still reached and still resolves.
* **(1) fires BEFORE training settles, deliberately.** MEASURED: 188-275 epochs to converge with it
  against 591-891 without.
* **(1) asks the gap's TREND, not one epoch's value.** A gap inflated by an unsettled network is
  falling, so it extrapolates below the bar and does not fire. MEASURED against an early gap inflated
  3x and 10x: converges 20/20 and at ``patience`` 32 waits the transient out, rounds stretching to
  97-122 epochs instead of growing the window every ``warmup + 1``.
* **Harmonic ``1/k`` weights.** A fixed window pins the effective sample size, ``sd(slope)`` freezes,
  and on an already-converged loss P reads 0.72-0.88 and never crosses 0.9 -- it never terminates.
  Geometric decay fails identically.
* **No validation plateau test.** If val is rising the gap grows and (1) adds data, which is correct;
  if val is still falling the gap is briefly overstated and data is added early, which wastes budget
  but cannot report a bad estimate.
* **``warmup_epochs`` may be small.** The posterior's own width holds the decision off.

VERIFIED across ``A/sqrt(t)``, ``A/t`` and exponential tails x patience 8/32 x 5 seeds: converges 30/30,
no hangs.

``iteration_limit`` is the per-design upper limit; exceeding it is a hard error.
The shared budget bounds the whole run: when the pool can't fit the next sample,
:meth:`train` returns ``None`` and the BO loop stops.

At a data addition the network is CARRIED by default -- same params, same optimiser state, a larger
window. ``reinit_on_grow`` rebuilds it instead and ``param_mix`` interpolates toward the
network the run started from (see ``__init__``); both default off. Neither is part of the procedure
above and neither may alter it.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ...utils.training import (bayesian_trend, masked_mean_sem,
                               probability_above, probability_change_below)
from .common import (Trainer, TrainResult, _round_down, window_sample_indices, fresh_design_network,
                     )

__all__ = ["DesignTrainer"]


class _DesignBase(Trainer):
    """The per-design training LOOP (window-growing convergence) + factory, shared by the per-design
    strategy (:class:`DesignTrainer`) and the continual strategy (:class:`ContinualTrainer`). The network
    LIFECYCLE (``_sample_indices`` / ``_init_design_network``) is ABSTRACT here -- each strategy IMPLEMENTS
    it, so neither overrides a concrete default."""

    @classmethod
    def from_config(cls, detector, config, *, checkpoint_dir=None, seed=0):
        """Build a trainer from a full run-config dict.

        ``detector`` is passed explicitly; the rest comes from ``config`` -- the
        ``training`` block (which holds the data, optimizer and convergence knobs)
        is spread onto the constructor; ``optimizer`` / ``device`` are built and
        ``regressor`` is passed through. ``checkpoint_dir`` and ``seed`` are
        run-level and supplied by the caller.
        """
        from ...utils.config import optimizer as make_optimizer, resolve_device

        training = {k: v for k, v in config["training"].items() if k != "optimizer"}
        return cls(
            detector,
            regressor_config=config["regressor"],
            optimizer=make_optimizer(config["training"]["optimizer"]),
            device=resolve_device(config.get("device")),
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            **training,
        )

    def __init__(
        self,
        detector,
        *,
        regressor_config: dict,
        optimizer: optax.GradientTransformation,
        batch: int,
        n0: int,
        n_increment: int,
        iteration_limit: int,
        warmup_epochs: int,
        patience: int,
        loss_precision: float,
        budget: int,
        reinit_on_grow: bool = False,
        param_mix: float = 0.0,
        val_fraction: float = 0.25,
        eval_batch: int | None = None,
        device=None,
        checkpoint_dir: str | None = None,
        seed: int = 0,
    ):
        self.n0 = int(n0)
        self.n_increment = int(n_increment)
        self.warmup_epochs = int(warmup_epochs)
        self.patience = int(patience)
        self.loss_precision = float(loss_precision)

        # WHAT HAPPENS TO THE NETWORK AT A DATA ADDITION. Both DEFAULT OFF, i.e. the original
        # behaviour -- the same params and the same optimiser state are carried into every window --
        # so a run that does not ask for either is unaffected by their existing. Neither touches a
        # termination path: they act AFTER `_sample_round` has already succeeded, and the loop still
        # exits only by convergence or by the window cap.
        #
        # `reinit_on_grow` rebuilds the network from scratch (params, buffer state AND optimiser
        # state) at every addition, holding the SCHEDULE fixed while varying the network -- the
        # complement of the schedule sweeps, which hold the network and vary the schedule.
        self.reinit_on_grow = bool(reinit_on_grow)
        # `param_mix` walks the SAME axis as `reinit_on_grow`, continuously: at a data addition the
        # parameters become `(1 - lambda) * current + lambda * INITIAL`, where INITIAL is the network
        # this run started from -- so 0 is the carried network and 1 rewinds to the start. It REPLACED an isotropic-kick knob (`param_noise`), which was withdrawn:
        #
        # It moves TOWARD A NETWORK rather than in a random direction, so it interpolates between the
        # two conditions actually measured (carry: 6/9 capped; rebuild: 0/9) instead of sampling an
        # isotropic ball whose largest tested radius, eps = 0.03, is ~47x short of the distance to a
        # fresh init (two independent draws at the same scale sit about `sqrt(2) * rms` apart).
        #
        # It RESETS THE OPTIMISER, as `reinit_on_grow` does (user, 2026-08-16): the moments are a
        # summary of the trajectory that produced the current parameters, and a rewind throws part of
        # that trajectory away, so carrying them forward would step the rewound network under second
        # moments it never earned. What remains between the two knobs is the network they move toward
        # -- `param_mix = 1.0` rewinds to the network THIS RUN STARTED FROM, `reinit_on_grow` draws a
        # NEW one -- and the buffer state, which only `reinit_on_grow` rebuilds.
        #
        # CONFOUND, stated because it is not removable by construction: if the current parameters and a
        # fresh draw have similar scale and are roughly independent, their average has ~0.71 of their
        # RMS. So a mix shrinks the network as well as moving it, and at lambda = 0.5 that shrink is
        # comparable to a strong weight decay applied once.
        self.param_mix = float(param_mix)
        if self.reinit_on_grow and self.param_mix > 0.0:
            raise ValueError("reinit_on_grow and param_mix both act at the SAME point in the loop (the "
                             "data addition) and are alternatives; pass at most one")

        # Window caps from this trainer's knobs: round the per-design cap so additions
        # land exactly on it (it also fixes the epoch length); the val cap mirrors the
        # train/val ratio. The base ``__init__`` then sizes the pools + builds kernels.
        iteration_limit = _round_down(self.n0, self.n_increment, int(iteration_limit))
        val_iteration_limit = max(int(batch), round(iteration_limit * val_fraction / (1.0 - val_fraction)))
        super().__init__(
            detector,
            regressor_config=regressor_config,
            optimizer=optimizer,
            batch=batch,
            budget=budget,
            iteration_limit=iteration_limit,
            val_iteration_limit=val_iteration_limit,
            val_fraction=val_fraction,
            eval_batch=eval_batch,
            device=device,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
        )

    def train(
        self,
        design_scaled,
        seed,
        *,
        init_params=None,
        on_epoch=None,
        step=0,
    ) -> TrainResult | None:
        """Train a regressor for one *scaled* design.

        Appends this design's events into the shared budget pools (a fresh window)
        and trains within that window. ``seed`` is an INT and drives everything random in this call --
        the network draw and the training keys -- so a design is reproducible from it alone and the
        driver need only replay its seed sequence to resume. ``init_params`` optionally warm-starts
        the params from a previously trained design (the buffer state stays fresh).

        Returns a :class:`TrainResult`, or ``None`` if the shared budget pool is
        exhausted (the design did not complete).
        """
        detector = self.detector
        design_scaled = np.asarray(design_scaled, dtype=np.float32)
        design = detector.to_nominal(design_scaled)  # physical Design namedtuple (what the pools store)
        design_phys = np.asarray(detector.flatten_design(design), dtype=np.float32)  # flat, for the checkpoint tree

        init_seq, training_seq = np.random.SeedSequence(int(seed)).spawn(2)

        # Network for this design (base: fresh / optionally warm-started; the
        # continual trainer keeps and continues the same one across designs).
        params, state, opt_state = self._init_design_network(init_seq, init_params)
        # The network this run STARTED from, kept for `param_mix`. Not a fresh draw: mixing toward the
        # run's own initial parameters is `initial + (1 - lambda) * (current - initial)`, i.e. a pure
        # SHRINK OF WHAT WAS LEARNED, with no new randomness introduced and no scale artefact (the two
        # are not independent, so their average does not lose RMS the way two independent draws would).
        initial_params = params

        tp, vp = self.train_pool, self.val_pool
        # This design's window starts at the current pool fill. The offsets handed
        # to the kernels are 0-d int32 jax.Array (dynamic -> no recompile).
        w0_train, w0_val = tp.current, vp.current
        w0_train_j, w0_val_j = jnp.int32(w0_train), jnp.int32(w0_val)

        manager = self._checkpoint_manager(step)
        design_tree = {"scaled": design_scaled, "physical": design_phys}

        train_loss_history, val_loss_history, pool_size_history = [], [], []
        train_sem_history, val_sem_history = [], []
        key = jax.random.PRNGKey(int(training_seq.generate_state(1)[0]))

        # Initial data (n0 <= iteration_limit, so this only fails on a full budget).
        if self._sample_round(design, w0_train, w0_val, self.n0) is None:
            return None

        round_start = 0  # history index where the current (post-add) round began
        epoch_in_round = 0
        objective = None  # (loss, std) set on convergence

        # Per-epoch callback (plotting) on a single background worker so its slow
        # I/O never stalls training; drained on every exit by the context manager.
        plot_ctx = ThreadPoolExecutor(max_workers=1) if on_epoch is not None else nullcontext()
        with plot_ctx as plot_pool:
            while True:
                train_count = tp.current - w0_train  # filled rows of the window
                val_count = vp.current - w0_val
                key, subkey = jax.random.split(key)
                params, state, opt_state, _ = self._train_epoch(
                    params,
                    state,
                    opt_state,
                    subkey,
                    w0_train_j,
                    jnp.int32(train_count),
                    tp.buffers(),
                )

                # Per-epoch losses: sequential masked passes over BOTH windows.
                train_eval = self._eval_train(params, state, tp.buffers(), w0_train_j)
                train_mean, train_sem = masked_mean_sem(train_eval, train_count)
                val_eval = self._eval_val(params, state, vp.buffers(), w0_val_j)
                val_mean, val_sem = masked_mean_sem(val_eval, val_count)
                train_mean = float(train_mean)
                train_sem = float(train_sem)
                val_mean = float(val_mean)
                val_sem = float(val_sem)

                train_loss_history.append(train_mean)
                val_loss_history.append(val_mean)
                train_sem_history.append(train_sem)
                val_sem_history.append(val_sem)
                pool_size_history.append(train_count)
                epoch_in_round += 1

                if on_epoch is not None:
                    plot_pool.submit(
                        on_epoch,
                        self._snapshot(
                            train_loss_history,
                            val_loss_history,
                            train_sem_history,
                            val_sem_history,
                            pool_size_history,
                            train_count,
                            val_count,
                        ),
                    )

                gap_signed = val_mean - train_mean  # val - train (signed)
                err = float(np.hypot(train_sem, val_sem))  # combined SEM of val-train
                diff = abs(gap_signed)

                # Warmup: unconditional training after every data addition.
                if epoch_in_round <= self.warmup_epochs:
                    continue

                # HISTORY = this round's POST-WARMUP epochs only; a data addition moves
                # `round_start`, which resets the fit, as it must -- the curve restarts.
                first = round_start + self.warmup_epochs
                tr = np.asarray(train_loss_history[first:], dtype=np.float64)
                va = np.asarray(val_loss_history[first:], dtype=np.float64)
                tr_s = np.asarray(train_sem_history[first:], dtype=np.float64)
                va_s = np.asarray(val_sem_history[first:], dtype=np.float64)
                if tr.shape[0] < 3:
                    continue  # two coefficients need three points before the posterior means anything
                # Prior: zero-mean Gaussian on both coefficients with 3*sigma = the highest loss at
                # the first post-warmup epoch -- weakly informative, scaled to this task's own loss.
                prior_sigma = max(float(tr[0]), float(va[0])) / 3.0
                # THE GAP IS `|val - train| + err` -- the spread of the two numbers PLUS the error of
                # their means, i.e. exactly the quantity reported as the objective's uncertainty. Not
                # `|val - train|` alone: a design that stopped on a small bias but a large statistical
                # error would hand BO a number whose stated spread exceeds the precision it claims.
                gap_sem = np.hypot(tr_s, va_s)
                gap_series = np.abs(va - tr) + gap_sem
                # Harmonic 1/k weighting on the FULL history: the effective sample size still grows, so
                # sd(slope) collapses and the test can fire, while a fixed window pins it and NEVER
                # terminates on an already-converged loss (measured P = 0.72-0.88 at 12-400 epochs).
                tr_mean, tr_cov = bayesian_trend(tr, tr_s, prior_sigma)
                gap_mean, gap_cov = bayesian_trend(gap_series, gap_sem, prior_sigma)

                # ========================================================================== #
                #  THE CONVERGENCE PROCEDURE -- SACRED. GIVEN BY PEOPLE WHO KNOW LOGIC.
                #  DO NOT EDIT, REORDER, EXTEND OR RETUNE. DO NOT ADD A BRANCH. DO NOT MAKE
                #  ANY PART OF IT CONFIGURABLE. There is one procedure and no way to select
                #  another. Full rationale, and the measurement behind every clause, in the
                #  module docstring at the top of this file.
                # ========================================================================== #
                #
                # THE CHECKS, IN ORDER.
                #
                # (1) P(gap at +patience > loss_precision) > 0.9 -> ADD DATA.
                #     An EARLY exit, deliberately before training has settled: a gap whose own trend
                #     says it will still exceed the precision a horizon from now is data starvation,
                #     and waiting for a plateau to establish that only spends epochs. This is the
                #     legacy rule 1 with the uncertainty accounted for -- rule 1 compares a single
                #     noisy epoch's gap to the bar and fires at the first legal epoch of every round
                #     (MEASURED: 127 consecutive rounds of exactly warmup+1 epochs), whereas this
                #     asks the gap's POSTERIOR whether it is going to stay above the bar.
                #
                # (2) P(train change over +patience < loss_precision/2) > 0.9 -> training is done.
                #     Then a DIRECT comparison of the gap decides between more data and returning.
                #
                # (3) otherwise train on.
                #
                # WHY (2.1) IS A COMPARISON AND NOT A PROBABILITY. An earlier version asked
                # `P(gap < precision) > 0.9` here, and it HUNG on every synthetic seed: the rule's own
                # dynamics drive the gap TO the precision, which is exactly where that probability
                # converges to ~0.5, so neither branch could fire however much evidence accumulated.
                # Ordering the checks removes the band -- when check (1) does not fire because
                # P is near 0.5, check (2) still resolves, because a measured value against a
                # threshold always resolves.
                p_gap_exceeds = probability_above(
                    gap_mean, gap_cov, self.patience, self.loss_precision, gap_series.shape[0]
                )
                if p_gap_exceeds > 0.9:
                    pass  # (1) starved -> fall through to the data addition below
                else:
                    p_train_settled = probability_change_below(
                        tr_mean, tr_cov, self.patience, 0.5 * self.loss_precision
                    )
                    if p_train_settled > 0.9:
                        if diff + err > self.loss_precision:
                            pass  # (2.1) settled but the gap is still wide -> add data
                        else:
                            # THE OBJECTIVE IS THE VALIDATION LOSS ALONE (user, 2026-08-14). It was
                            # `(train + val)/2`, half of which is training loss -- so any intervention
                            # that trades training fit for generalisation was charged half its benefit
                            # as a cost, and anything that overfit harder was rewarded. MEASURED: it
                            # inflated every regulariser's apparent price by about 2x, and it hid a
                            # train-down/validation-up divergence under the growth schedule. The
                            # THE OBJECTIVE IS THE MIDPOINT (user, 2026-08-18), reverting the
                            # validation-only report. The estimand is the loss in the limit of
                            # INFINITE TRAINING DATA, which is a property of the DESIGN. At finite
                            # data the two channels straddle it -- train below, because the network
                            # fitted its own sample; val above, because the network is short of the
                            # limit -- so `L_inf` lies inside `[train, val]` and, under ignorance of
                            # where, the uniform model over that interval is the honest one: mean at
                            # the midpoint, standard deviation `(val - train) / sqrt(12)`.
                            #
                            # THE NOISE IS THE UNIFORM SPREAD AND THE MIDPOINT'S OWN SAMPLING
                            # ERROR, IN QUADRATURE. The estimator is the midpoint, so its sampling
                            # error is `0.5 * hypot(train_sem, val_sem)` -- HALF `err`, because
                            # `Var[(a+b)/2] = (var_a + var_b)/4`; charging it the full `err` would be
                            # the error of the DIFFERENCE, not of the average. The two terms are
                            # independent -- where `L_inf` sits inside the bracket has nothing to do
                            # with the noise in locating the bracket's ends -- so they add in
                            # variance, not linearly.
                            # `scripts/bo.py` passes this straight to the GP as that observation's
                            # noise (`bo_opt.append(..., noise=result.objective_std)`), which the GP
                            # takes as a per-observation standard deviation and does not fit.
                            #
                            # NOTHING ABOVE CHANGES: the two exit clauses and every `_sample_round`
                            # decision are untouched. Only what is REPORTED changes, never a decision.
                            objective = (
                                0.5 * (train_mean + val_mean),
                                float(np.hypot(diff / np.sqrt(12.0), 0.5 * err)),
                            )  # (2.2)
                            print(
                                f"  [converged/bayes] train={train_mean:.4f} val={val_mean:.4f} "
                                f"diff={diff:.4f} err={err:.4f} prec={self.loss_precision:.4f} | "
                                f"P(gap>LP)={p_gap_exceeds:.3f} P(settled)={p_train_settled:.3f} "
                                f"| window={train_count}"
                            )
                            break
                    else:
                        continue  # (3) training has not finished -> keep training
                # CHECK (1) or (2.1) fired -> add data.
                n_added = self._sample_round(
                    design,
                    w0_train,
                    w0_val,
                    self.n_increment,
                )
                if n_added is None:
                    return None  # shared budget pool exhausted mid-design
                if n_added == 0:
                    raise RuntimeError(
                        f"design did not reach precision within iteration_limit "
                        f"{self.iteration_limit} (window={train_count}); "
                        f"diff={diff:.4g} err={err:.4g} diff+err={diff + err:.4g} vs "
                        f"precision={self.loss_precision:.4g}. "
                        f"A healthy network must reach precision within iteration_limit "
                        f"-- raise it or fix model/regularisation."
                    )
                # THE NETWORK AT THE ADDITION. Default: nothing happens here at all -- the same params
                # and the same optimiser state carry into the larger window, which is what "continued"
                # means and is the thing under test. Both branches run only AFTER the addition has
                # succeeded, so neither can remove the loop's exits.
                if self.reinit_on_grow:
                    # A NEW network by this strategy's own definition (per-design: fresh params, fresh
                    # buffer state, fresh optimiser state -- a carried Adam moment is part of what is
                    # being continued, so it goes too). Each round draws its own child sequence, so the
                    # rounds are independent inits rather than the same one repeated; the FIRST network
                    # is untouched, since nothing is spawned before it is built.
                    params, state, opt_state = self._init_design_network(init_seq.spawn(1)[0], None)
                elif self.param_mix > 0.0:
                    # Toward the network THIS RUN WAS INITIALISED WITH, not toward a new random draw.
                    # `params = initial + (1 - lambda) * (current - initial)`: it discards a fraction of
                    # the accumulated fit and nothing else. A fresh draw would confound "throw away what
                    # was learned" with "land in a different random basin", and would inject randomness
                    # that differs between two arms whose whole point is a matched initialisation.
                    # lambda = 1 rewinds to the initial network exactly; the buffer state is left alone.
                    # THE OPTIMISER IS RESET with the rewind: Adam's moments describe the trajectory
                    # that reached the current parameters, and the rewind discards a fraction of that
                    # trajectory, so they no longer describe the network they would be stepping.
                    mix = self.param_mix
                    params = jax.tree.map(lambda p, q: q + (1.0 - mix) * (p - q), params, initial_params)
                    opt_state = self.optimizer.init(params)
                round_start = len(train_loss_history)
                epoch_in_round = 0
                print(f"  [grow] window -> {tp.current - w0_train}, pool {tp.current}/{tp.capacity}")
                continue

            # ONE checkpoint per design, written at convergence. Saving every epoch wrote 27 files
            # an epoch to retain only the last three (96% of them deleted again), and orbax's
            # async save blocks the NEXT one until its background write lands -- free only while
            # the epoch outlasts the write (~200 ms locally, more on network storage). Nothing
            # read the earlier epochs anyway: both consumers take ``latest_step()``, i.e. exactly
            # this final state (verify_trajectory._restore_design_network, continue_reported).
            self._save_checkpoint(
                manager,
                len(train_loss_history),
                params,
                state,
                design_tree,
                train_mean,
                val_mean,
            )
            if manager is not None:
                manager.wait_until_finished()
            self._persist_network(params, state, opt_state)  # continual: keep it
            objective_loss, objective_std = objective
            spent = (tp.current - w0_train) + (vp.current - w0_val)
            return TrainResult(objective_loss, objective_std, spent, params)


class DesignTrainer(_DesignBase):
    """The standard per-design strategy: a FRESH network per design, minibatches drawn uniformly over the
    current window (history-ignoring)."""

    def _sample_indices(self, key, start, count):
        return window_sample_indices(self, key, start, count)

    def _sample_weights(self):
        return None  # every row is the current design's; nothing to weight

    def _init_design_network(self, init_seq, init_params):
        return fresh_design_network(self, init_seq, init_params)

    def _carried_state(self):
        return {}  # nothing survives a design boundary: every design gets its own network

    def _load_carried_state(self, data):
        """Nothing to load -- see :meth:`_carried_state`."""
