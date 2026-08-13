"""Seed propagation + advancement across the BO / training pipeline.

Guarantees that (a) the same seed reproduces a run exactly, (b) a different seed
changes the result, and (c) the RNG *advances* within a run (BO proposals differ,
trainer init/data/training keys are decorrelated). These guard the spawn / split
chains in ``BayesianOptimizer`` and the trainers against silent seed reuse.
"""

import numpy as np
import jax
import optax

from detopt.bo import BayesianOptimizer
from analytic import analytic_detector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer, FullBudgetTrainer

_GP = dict(
    n_folds=4,
    n_restarts=2,
    n_steps=20,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI = dict(n_restarts=4, n_steps=20)


def _bo(seed, d=3, n_init=4):
    return BayesianOptimizer(d, gp=_GP, ei=_EI, n_init=n_init, seed=seed)


def _proposal_seq(seed, n=4):
    bo = _bo(seed)
    return [np.asarray(bo.propose()) for _ in range(n)]


def test_bo_init_proposals_advance_and_reproduce():
    """Random-init proposals differ call-to-call (key advances), yet a fixed seed
    replays the exact sequence and a new seed changes it."""
    seq_a = _proposal_seq(7)
    seq_b = _proposal_seq(7)
    seq_c = _proposal_seq(8)
    # Advancement: no two consecutive proposals coincide.
    for i in range(len(seq_a) - 1):
        assert not np.allclose(seq_a[i], seq_a[i + 1])
    # Reproducible under the same seed; sensitive to the seed.
    assert all(np.allclose(a, b) for a, b in zip(seq_a, seq_b))
    assert not np.allclose(seq_a[0], seq_c[0])


def test_bo_gp_proposal_reproducible():
    """The GP+EI proposal (post-init) is reproducible for a fixed seed; the fitting
    / acquisition keys are threaded from the BO seed, not hardcoded."""

    def run(seed):
        bo = _bo(seed, n_init=4)
        rng = np.random.default_rng(0)  # fixed data, so only the BO seed varies
        for _ in range(6):
            x = rng.uniform(-3.0, 3.0, size=3).astype("float32")
            bo.append(x, float(np.sum(x**2)), noise=1e-2)
        return np.asarray(bo.propose())

    assert np.allclose(run(3), run(3))
    assert not np.allclose(run(3), run(4))


def _design_trainer(seed):
    det = analytic_detector()
    return DesignTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.1}},
        optimizer=optax.adamw(1e-3, weight_decay=1e-3),
        batch=64,
        n0=256,
        n_increment=128,
        iteration_limit=256,
        warmup_epochs=2,
        patience=3,
        loss_precision=0.5,
        budget=8000,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        checkpoint_dir=None,
        seed=seed,
    )


def _loss(trainer, train_seed):
    det = analytic_detector()
    design = np.zeros(det.design_dim(), dtype=np.float32)
    return float(trainer.train(design, np.random.SeedSequence(train_seed)).objective_loss)


def test_design_trainer_reproducible_by_train_seed():
    """DesignTrainer init/data/training all come from the per-design ``seed_seq``:
    same seq reproduces the loss exactly, a different seq changes it."""
    a = _loss(_design_trainer(0), train_seed=0)
    b = _loss(_design_trainer(0), train_seed=0)
    c = _loss(_design_trainer(0), train_seed=1)
    assert abs(a - b) < 1e-6  # fully reproducible
    assert abs(a - c) > 1e-3  # the train seed_seq actually drives init + data


def _continual(seed):
    det = analytic_detector()
    return ContinualTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.1}},
        optimizer=optax.adamw(1e-3, weight_decay=1e-3),
        batch=64,
        n0=256,
        n_increment=128,
        iteration_limit=256,
        warmup_epochs=2,
        patience=3,
        loss_precision=0.5,
        budget=8000,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        seed=seed,
    )


def test_continual_trainer_seed_stored_and_propagated():
    """The persistent network's init seed is the stored run seed (not kwargs.get):
    same constructor seed -> identical result, different -> different net -> different."""
    ca, cb, cc = _continual(0), _continual(0), _continual(5)
    assert ca.seed == 0 and cc.seed == 5  # stored, not defaulted
    # The property that matters is that the seed reaches the PERSISTENT NETWORK's initialisation.
    # Asserting only that two runs differ does NOT test it: `Trainer.__init__` also seeds
    # `shuffled_event_index`, so the train/val split alone moves the loss by ~0.05 even when the two
    # networks are byte-identical -- verified by hardcoding the net's seed, under which every other
    # assertion in this test still passes. Compare the initial weights directly.
    assert any(
        not np.allclose(x, y)
        for x, y in zip(jax.tree.leaves(ca._running[0]), jax.tree.leaves(cc._running[0]))
    )
    assert all(
        np.allclose(x, y)
        for x, y in zip(jax.tree.leaves(ca._running[0]), jax.tree.leaves(cb._running[0]))
    )
    a = _loss(ca, train_seed=0)
    b = _loss(cb, train_seed=0)
    c = _loss(cc, train_seed=0)
    assert abs(a - b) < 1e-6  # reproducible
    assert abs(a - c) > 1e-3  # constructor seed -> persistent-net init -> result


def _fullbudget(seed):
    det = analytic_detector()
    return FullBudgetTrainer(
        det,
        regressor_config={"set-regressor": {"features": [[16, 16]], "p_dropout": 0.1}},
        optimizer_config={"adamw": {"learning_rate": 1e-3, "weight_decay": 1e-3}},
        batch=64,
        budget=4000,
        max_epochs=2,
        val_fraction=0.25,
        eval_batch=128,
        device=None,
        seed=seed,
    )


def test_fullbudget_reproducible_by_seed():
    """FullBudgetTrainer (fresh init + full-budget fill) is reproducible per seed."""
    a = _loss(_fullbudget(0), train_seed=0)
    b = _loss(_fullbudget(0), train_seed=0)
    c = _loss(_fullbudget(0), train_seed=1)
    assert abs(a - b) < 1e-6
    assert abs(a - c) > 1e-3
