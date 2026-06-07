"""Tests for detopt.nn.trainer.DesignTrainer (convergence procedure + dropout)."""

import numpy as np
import jax.numpy as jnp
import pytest
from flax import nnx

import jax

from detopt.detector.debug import DebugDetector
from detopt.nn import from_config
from detopt.nn.trainer import ContinualTrainer, DesignTrainer, TrainResult


def _adam(lr=1e-3):
    import optax

    return optax.adamw(learning_rate=lr, weight_decay=1e-3)


def test_dropout_active_only_when_training(seed):
    """deterministic=True -> dropout OFF (identical); False -> ON (outputs differ)."""
    det = DebugDetector()
    reg = from_config(
        det,
        config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.5}},
        rngs=nnx.Rngs(seed),
    )
    rng = np.random.default_rng(seed)
    B, M, F = 4, det.max_hits_per_event, det.combined_feature_dim
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


def _small_trainer(loss_precision, *, n0, n_increment, iteration_limit, budget, seed):
    det = DebugDetector()
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
        flatness_tol=1e-1,
        loss_precision=loss_precision,
        budget=budget,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        checkpoint_dir=None,
        seed=seed,
    )


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
    result = trainer.train(np.zeros(det.design_dim(), dtype=np.float32), np.random.SeedSequence(seed))
    assert isinstance(result, TrainResult)
    assert np.isfinite(result.objective_loss) and result.objective_loss > 0
    # Accept invariant: the loss estimate's own SEM is within precision.
    assert result.objective_std <= loss_precision + 1e-6
    assert 0 < result.spent
    assert result.params is not None and result.state is not None


def test_design_trainer_crashes_when_iteration_limit_exceeded(seed):
    """An unreachable precision must crash loudly at iteration_limit, never silently."""
    det, trainer = _small_trainer(1e-9, n0=64, n_increment=64, iteration_limit=128, budget=100_000, seed=seed)
    with pytest.raises(RuntimeError, match="did not reach precision within iteration_limit"):
        trainer.train(np.zeros(det.design_dim(), dtype=np.float32), np.random.SeedSequence(seed))


def test_design_trainer_returns_none_when_budget_too_small(seed):
    """If the budget pool can't fit the initial sample, return None (not crash)."""
    det, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=10, seed=seed)
    result = trainer.train(np.zeros(det.design_dim(), dtype=np.float32), np.random.SeedSequence(seed))
    assert result is None


def test_pool_accumulates_across_designs(seed):
    """Designs append into the shared budget pool (windows accumulate)."""
    det, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=seed)
    design = np.zeros(det.design_dim(), dtype=np.float32)
    assert trainer.train(design, np.random.SeedSequence(seed)) is not None
    after_first = trainer.train_pool.n_current
    assert trainer.train(design, np.random.SeedSequence(seed + 1)) is not None
    after_second = trainer.train_pool.n_current
    assert 0 < after_first < after_second  # the second design appended more data


def _small_continual(loss_precision, *, n0, n_increment, iteration_limit, budget, seed):
    det = DebugDetector()
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
        flatness_tol=1e-1,
        loss_precision=loss_precision,
        budget=budget,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        seed=seed,
    )


def test_continual_trainer_persists_and_replays(seed):
    """The continual trainer keeps one network and accumulates a replay history."""
    det, trainer = _small_continual(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=seed)
    design = np.zeros(det.design_dim(), dtype=np.float32)
    assert trainer._running is None
    assert trainer.train(design, np.random.SeedSequence(seed)) is not None
    assert trainer._running is not None  # network persisted across the call
    after_first = trainer.train_pool.n_current
    # The next design continues the SAME network (init returns the persisted tuple).
    assert trainer._init_design_network(np.random.SeedSequence(0), None, None) is trainer._running
    # ... and appends to the pool, so history is available for replay.
    assert trainer.train(design, np.random.SeedSequence(seed + 1)) is not None
    assert trainer.train_pool.n_current > after_first


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
    """No design scramble: the design fed to ``combine`` reaches the network, so the
    shared loss kernel (used by every trainer) gives a different loss for the real
    encoded design than for a zeroed one."""
    det = DebugDetector()
    _, trainer = _small_trainer(0.5, n0=256, n_increment=128, iteration_limit=512, budget=20_000, seed=0)
    reg_def, params, state = trainer._build_regressor(0)
    loss_fn = trainer._make_loss_fn(reg_def)

    B = 16
    design = det.get_current_design_array()
    phys = np.broadcast_to(design[None, :], (B, det.design_dim()))
    _gt, X, mask, targets = det(np.random.SeedSequence(0), phys)
    enc = np.asarray(det.encode_design(design), np.float32)
    enc_b = np.broadcast_to(enc[None, :], (B, det.design_dim())).astype(np.float32)

    key = jax.random.PRNGKey(0)
    X, mask, targets = jnp.asarray(X), jnp.asarray(mask), jnp.asarray(targets)
    loss_real, _ = loss_fn(params, state, key, X, mask, jnp.asarray(enc_b), targets)
    loss_zero, _ = loss_fn(params, state, key, X, mask, jnp.zeros_like(jnp.asarray(enc_b)), targets)
    assert abs(float(loss_real) - float(loss_zero)) > 1e-4  # the design actually feeds combine
