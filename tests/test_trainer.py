"""Tests for detopt.nn.trainer.DesignTrainer (convergence procedure + dropout)."""

import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

import jax

from analytic import analytic_detector, DESIGN
from detopt.nn import from_config
from detopt.nn.trainer import ContinualTrainer, DesignTrainer, TrainResult


# Test design (the detector holds none): flat [station_z(4), view_tilt(4), field_strength].
# FairShip-matched geometry: SST stations at 8407, 8607 | 9307, 9507; nominal B ~ 0.178 T.
_DEBUG_DESIGN = DESIGN


def _adam(lr=1e-3):
    import optax

    return optax.adamw(learning_rate=lr, weight_decay=1e-3)


def test_dropout_active_only_when_training(seed):
    """deterministic=True -> dropout OFF (identical); False -> ON (outputs differ)."""
    det = analytic_detector()
    reg = from_config(
        det,
        config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.5}},
        rngs=nnx.Rngs(seed),
    )
    rng = np.random.default_rng(seed)
    B, M, F = 4, det.max_hits_per_event, det.combined_feature_dim()
    features = jnp.asarray(rng.standard_normal((B, M, F)), dtype=jnp.float32)
    mask = jnp.ones((B, M), dtype=jnp.int32)

    # Dropout off: two evaluation passes are bit-identical.
    a = reg(features, mask, deterministic=True)
    b = reg(features, mask, deterministic=True)
    assert np.allclose(np.asarray(a), np.asarray(b))

    # Dropout on: two training passes differ (the rng advances each call).
    c = reg(features, mask, deterministic=False)
    d = reg(features, mask, deterministic=False)
    assert not np.allclose(np.asarray(c), np.asarray(d))


def _small_trainer(loss_precision, *, n0, n_increment, iteration_limit, budget, seed, checkpoint_dir=None,
                   reveal=None):
    det = analytic_detector()
    return det, DesignTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.1}},
        optimizer=_adam(),
        batch=64,
        n0=n0,
        n_increment=n_increment,
        iteration_limit=iteration_limit,
        warmup_epochs=2,
        patience=3,
        loss_precision=loss_precision,
        budget=budget,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        reveal=reveal,
    )


def test_warm_start_reads_the_design_checkpoint(seed, tmp_path):
    """A warm start loads a design's CHECKPOINT, and it is the network that design ended on.

    ``scripts/bo.py`` keeps no historical parameters: ``continue`` and ``closest`` choose a design
    NUMBER and the trainer reads that design's checkpoint back, so the pool a warm start may draw from
    is everything the run has measured rather than everything this process still holds -- which is what
    lets a resumed run warm-start from designs scored before the interruption. The identity that makes
    the two equivalent is asserted here: what comes back off disk is what ``train`` returned, leaf for
    leaf. A missing checkpoint must RAISE, never silently degrade to a cold start."""
    det, trainer = _small_trainer(
        0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=seed, checkpoint_dir=str(tmp_path)
    )
    design = np.zeros(det.design_dim(), dtype=np.float32)
    result = trainer.train(design, seed, step=0)
    assert result is not None

    restored = trainer.restore_design_parameters(0)
    trained, back = jax.tree.leaves(result.params), jax.tree.leaves(restored)
    assert len(trained) > 0 and len(trained) == len(back)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(trained, back))

    with pytest.raises(FileNotFoundError):
        trainer.restore_design_parameters(1)


def test_design_trainer_converges(seed):
    """A loose precision converges and reports a precise, finite loss estimate."""
    loss_precision = 0.5
    det, trainer = _small_trainer(
        loss_precision,
        n0=256,
        n_increment=128,
        iteration_limit=512,
        budget=20_000,
        seed=seed,
    )
    result = trainer.train(np.zeros(det.design_dim(), dtype=np.float32), seed)
    assert isinstance(result, TrainResult)
    assert np.isfinite(result.objective_loss) and result.objective_loss > 0
    # Accept invariant: the loss estimate's own SEM is within precision.
    assert result.objective_std <= loss_precision + 1e-6
    assert 0 < result.spent
    assert result.params is not None


def test_design_trainer_crashes_when_iteration_limit_exceeded(seed):
    """An unreachable precision must crash loudly at iteration_limit, never silently."""
    det, trainer = _small_trainer(1e-9, n0=64, n_increment=64, iteration_limit=128, budget=100_000, seed=seed)
    with pytest.raises(RuntimeError, match="did not reach precision within iteration_limit"):
        trainer.train(np.zeros(det.design_dim(), dtype=np.float32), seed)


def test_design_trainer_returns_none_when_budget_too_small(seed):
    """If the budget pool can't fit the initial sample, return None (not crash)."""
    det, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=10, seed=seed)
    result = trainer.train(np.zeros(det.design_dim(), dtype=np.float32), seed)
    assert result is None


def test_pool_accumulates_across_designs(seed):
    """Designs append into the shared budget pool (windows accumulate)."""
    det, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=seed)
    design = np.zeros(det.design_dim(), dtype=np.float32)
    assert trainer.train(design, seed) is not None
    after_first = trainer.train_pool.current
    assert trainer.train(design, seed + 1) is not None
    after_second = trainer.train_pool.current
    assert 0 < after_first < after_second  # the second design appended more data


def _small_continual(loss_precision, *, n0, n_increment, iteration_limit, budget, seed):
    det = analytic_detector()
    return det, ContinualTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.1}},
        optimizer=_adam(),
        batch=64,
        n0=n0,
        n_increment=n_increment,
        iteration_limit=iteration_limit,
        warmup_epochs=2,
        patience=3,
        loss_precision=loss_precision,
        budget=budget,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        seed=seed,
    )


def test_continual_trainer_persists_and_replays(seed):
    """The continual trainer keeps ONE network across designs and accumulates a replay history.

    The persistent net is built EAGERLY in ``__init__`` -- never lazily on the first ``train()`` --
    so no jax array is ever cached behind a ``None``. What "persists" therefore means here is that
    the same network is CONTINUED rather than rebuilt: the weights that come out of one design are
    the weights that go into the next."""
    det, trainer = _small_continual(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=seed)
    design = np.zeros(det.design_dim(), dtype=np.float32)
    assert trainer._running is not None  # eager: the network exists before any training
    before = jax.tree.map(np.asarray, trainer._running[0])

    assert trainer.train(design, seed) is not None
    after_first = trainer.train_pool.current
    trained = jax.tree.map(np.asarray, trainer._running[0])
    # Training WROTE BACK to the persistent tuple: at least one leaf moved.
    assert any(not np.allclose(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(trained)))
    # The next design continues that same net -- params and buffers are handed back UNCHANGED -- but
    # the OPTIMISER IS RESTARTED, so Adam's moments never cross a design boundary. That is what makes
    # the carried state complete and a resumed run exact, so it is asserted rather than assumed.
    next_params, next_state, next_opt = trainer._init_design_network(np.random.SeedSequence(0), None)
    assert next_params is trainer._running[0] and next_state is trainer._running[1]
    assert all(np.allclose(np.asarray(leaf), 0.0) for leaf in jax.tree.leaves(next_opt))
    # ... and appends to the pool, so history is available for replay.
    assert trainer.train(design, seed + 1) is not None
    assert trainer.train_pool.current > after_first


def test_continual_replay_sampling():
    """Half the batch comes from the current window, half from past history."""
    import jax.numpy as jnp

    _, trainer = _small_continual(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0)
    half = trainer.batch // 2
    key = jax.random.PRNGKey(0)
    # With history (start>0): current half in [start, start+count), past in [0, start).
    idx = np.asarray(trainer._sample_indices(key, jnp.int32(1000), jnp.int32(200)))
    assert idx.shape == (trainer.batch,)
    cur, past = idx[: trainer.batch - half], idx[trainer.batch - half :]
    assert (cur >= 1000).all() and (cur < 1200).all()
    assert (past >= 0).all() and (past < 1000).all()
    # No history (start==0): both halves drawn from the current window [0, count).
    idx0 = np.asarray(trainer._sample_indices(key, jnp.int32(0), jnp.int32(200)))
    assert (idx0 >= 0).all() and (idx0 < 200).all()


def test_trainers_are_design_conditioned():
    """A trainer that REVEALS the design feeds it to ``combine``, so the shared loss kernel gives a
    different loss for the real design than for a shifted one. The pool stores raw events + raw
    physical design; ``combine`` encodes it.

    ⚠️ WHICH ARMS REVEAL IT IS A PER-STRATEGY DEFAULT (``Trainer.default_reveal``): the
    per-design arms withhold unless ``training.reveal`` says otherwise, so this asks for it
    explicitly rather than assuming every trainer is design-conditioned."""
    det = analytic_detector()
    _, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0,
                                reveal='design')
    reg_def, params, state = trainer._build_regressor(0)
    loss_fn = trainer._make_loss_fn(reg_def)

    B = 16
    design = _DEBUG_DESIGN
    phys = np.broadcast_to(design[None, :], (B, det.design_dim())).astype(np.float32)
    _gt, event, mask, target = det(phys, np.arange(B))

    key = jax.random.PRNGKey(0)
    design_real = jnp.asarray(phys)  # per-event PHYSICAL design (combine encodes it)
    design_alt = design_real.at[:, : det.n_stations].add(40.0)  # shift station z -> different geometry
    loss_real, _ = loss_fn(params, state, key, event, mask, design_real, target)
    loss_alt, _ = loss_fn(params, state, key, event, mask, design_alt, target)
    assert abs(float(loss_real) - float(loss_alt)) > 1e-4  # the design actually feeds combine


def test_zero_design_keeps_the_shape_and_removes_the_information():
    """``reveal='zeros'`` is the CAPACITY-MATCHED control: the design is still revealed, so the
    features keep their full width and the regressor its full input, but the values handed to
    ``combine`` are zeros. That is a different thing from ``reveal='none'``, which narrows the input
    instead."""
    det = analytic_detector()
    _, sighted = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0,
                                reveal='design')
    _, zeroed = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0,
                               reveal='zeros')

    B = 16
    phys = np.broadcast_to(_DEBUG_DESIGN[None, :], (B, det.design_dim())).astype(np.float32)
    _gt, event, mask, _target = det(phys, np.arange(B))
    design = jnp.asarray(phys)

    with_design = sighted._combine(event, design, mask)
    with_zeros = zeroed._combine(event, design, mask)
    # same shape -- the regressor is unchanged, only the values it reads are
    assert with_design.shape == with_zeros.shape
    assert float(jnp.max(jnp.abs(with_design - with_zeros))) > 1e-6

    # and it really is the ZERO design, not some other one
    straight = det.combine(event, jnp.zeros_like(design), mask=mask)
    assert float(jnp.max(jnp.abs(with_zeros - straight))) == 0.0

    # two DIFFERENT designs are indistinguishable to a zero-design arm
    other = design + 40.0
    assert float(jnp.max(jnp.abs(zeroed._combine(event, other, mask) - with_zeros))) == 0.0
    assert float(jnp.max(jnp.abs(sighted._combine(event, other, mask) - with_design))) > 1e-6


def test_unknown_reveal_is_refused():
    """``reveal`` names one of the three layouts in ``REVEAL``. Anything else is refused at
    construction rather than falling through to the strategy default, which would silently train a
    different arm than the one the config asked for -- the old two-flag spelling
    (``reveal_design`` + ``zero_design``) is exactly the sort of stale value this catches."""
    with pytest.raises(ValueError, match='reveal must be one of'):
        _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0,
                       reveal='zero_design')
