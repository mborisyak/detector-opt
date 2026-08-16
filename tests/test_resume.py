"""Resume state: what ``scripts/bo.py`` writes between designs, and that reading it back is exact.

The driver restarts an interrupted run at the DESIGN BOUNDARY -- the design it died on is proposed and
trained again from scratch -- so nothing mid-design is kept. What must survive is everything that
crosses a boundary and cannot be recomputed without re-spending the detector budget: the optimiser's
evidence, the event pools (CONTENTS AND CURSORS), and the continual strategy's persistent network.

These tests use the ``linear`` debug detector: two probe positions on ``y = w x + b + noise``, no
physics and no data file, so a pool round-trip costs milliseconds. The seed-sequence half of the
contract -- that replaying a branch k times lands on the same stream -- is checked here too, because
it is what makes "resume" mean "the same run" rather than "a run that continues".
"""

import numpy as np
import optax
import pytest

import jax

from detopt.bo import BayesianOptimizer
from detopt.detector import LinearDetector
from detopt.nn.trainer import ContinualTrainer, DesignTrainer
from detopt.utils import io

_GP = dict(
    n_folds=4,
    n_restarts=2,
    n_steps=20,
    log_lengthscale_prior_bounds=(-2.0, 2.0),
    log_amplitude_prior_bounds=(-2.0, 2.0),
)
_EI = dict(n_restarts=4, n_steps=20)


def _optimiser(d=2, n_init=4):
    return BayesianOptimizer(d, gp=_GP, ei=_EI, n_init=n_init)


def _trainer(cls, seed, *, budget=8192, **kwargs):
    detector = LinearDetector()
    return detector, cls(
        detector,
        regressor_config={"set-regressor": {"features": [[16, 16]]}},
        optimizer=optax.adamw(1e-3, weight_decay=1e-3),
        batch=64,
        n0=512,
        n_increment=256,
        iteration_limit=1024,
        warmup_epochs=2,
        patience=3,
        loss_precision=0.5,
        budget=budget,
        val_fraction=0.25,
        eval_batch=256,
        device=None,
        seed=seed,
        **kwargs,
    )


# ---------------------------------------------------------------------- #
# The commit protocol
# ---------------------------------------------------------------------- #
def test_commit_publishes_the_whole_set_or_nothing(tmp_path):
    """Staged files are invisible until ``commit``, and then all of them are."""
    first, second = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    io.stage(first, {"x": np.arange(3)})
    io.stage(second, {"x": np.arange(4)})
    assert io.restore_path(first) is None and io.restore_path(second) is None
    io.commit([first, second])
    assert io.restore_path(first) == first and io.restore_path(second) == second


def test_commit_refuses_a_partial_set(tmp_path):
    """A path with nothing staged is an error, not a silently skipped file -- otherwise a caller that
    forgot one object would publish a mixed generation."""
    first, second = str(tmp_path / "a.npz"), str(tmp_path / "b.npz")
    io.stage(first, {"x": np.arange(3)})
    with pytest.raises(FileNotFoundError):
        io.commit([first, second])


def test_interrupted_commit_falls_back_to_the_previous_generation(tmp_path):
    """A leftover ``.old`` with no file in place is the rename-aside having happened and the
    rename-into-place not: the previous generation is still the last consistent state."""
    path = str(tmp_path / "state.npz")
    io.atomic_save({path: {"x": np.arange(3)}})
    import os

    os.replace(path, path + ".old")  # a commit cut between step 1 and step 2
    assert io.restore_path(path) == path + ".old"


# ---------------------------------------------------------------------- #
# BayesianOptimizer
# ---------------------------------------------------------------------- #
def test_optimizer_round_trip_reproduces_the_next_proposal(tmp_path):
    """Evidence + initial block restore exactly, so the restored optimiser proposes what the original
    would have proposed next."""
    path = str(tmp_path / "optimizer.npz")
    rng = np.random.default_rng(0)
    original = _optimiser()
    for step in range(7):
        x = np.asarray(original.propose(100 + step), np.float32)
        original.append(x, float(np.sum(x**2)), noise=1e-2)
    expected = np.asarray(original.propose(999))
    original.persist(path)
    io.commit([path])

    restored = _optimiser()
    restored.restore(path)
    assert np.allclose(restored.X, original.X) and np.allclose(restored.y, original.y)
    assert np.allclose(np.asarray(restored.propose(999)), expected)


def test_optimizer_restore_keeps_the_initial_block(tmp_path):
    """A run interrupted DURING the Sobol block continues the same block, not a fresh one -- the
    low-discrepancy guarantee is a property of the set, so a redraw would break it."""
    path = str(tmp_path / "optimizer.npz")
    original = _optimiser(n_init=4)
    first = np.asarray(original.propose(7))
    original.append(first, 1.0, noise=1e-2)
    original.persist(path)
    io.commit([path])

    restored = _optimiser(n_init=4)
    restored.restore(path)
    # The seed of this call is IGNORED while the block is being consumed; row 1 must come back.
    assert np.allclose(np.asarray(restored.propose(123456)), np.asarray(original.propose(123456)))


def test_optimizer_restore_refuses_a_mismatched_shape(tmp_path):
    """State written at another design dimension is not silently loaded."""
    path = str(tmp_path / "optimizer.npz")
    _optimiser(d=2).persist(path)
    io.commit([path])
    with pytest.raises(ValueError, match="state is d="):
        _optimiser(d=3).restore(path)


# ---------------------------------------------------------------------- #
# Trainers
# ---------------------------------------------------------------------- #
def test_trainer_round_trip_restores_pool_contents_and_cursors(tmp_path):
    """A resumed trainer holds the same events at the same fill, so the next design's window opens
    where it would have and consumes the event index it would have."""
    path = str(tmp_path / "trainer.npz")
    detector, original = _trainer(DesignTrainer, seed=3)
    design = np.full(detector.design_dim(), 0.5, np.float32)
    assert original.train(design, 11) is not None
    original.persist(path)
    io.commit([path])

    _, restored = _trainer(DesignTrainer, seed=3)
    restored.restore(path)
    for a, b in ((original.train_pool, restored.train_pool), (original.val_pool, restored.val_pool)):
        assert a.current == b.current > 0
        for x, y in zip(jax.tree.leaves(a.buffers()), jax.tree.leaves(b.buffers())):
            assert np.array_equal(np.asarray(x), np.asarray(y))


def test_trainer_resumes_onto_the_same_trajectory(tmp_path):
    """The whole point: a trainer restored mid-run trains the NEXT design to the same number as one
    that was never interrupted. Both are handed the same per-iteration seed, as the driver does."""
    path = str(tmp_path / "trainer.npz")
    detector, reference = _trainer(DesignTrainer, seed=3)
    design_a = np.full(detector.design_dim(), 0.5, np.float32)
    design_b = np.array([0.0, 1.0], np.float32)
    assert reference.train(design_a, 11) is not None
    reference.persist(path)
    io.commit([path])
    uninterrupted = reference.train(design_b, 12)

    _, resumed = _trainer(DesignTrainer, seed=3)
    resumed.restore(path)
    after_resume = resumed.train(design_b, 12)
    assert after_resume.objective_loss == pytest.approx(uninterrupted.objective_loss, abs=1e-6)
    assert after_resume.spent == uninterrupted.spent


def test_trainer_restore_refuses_another_seed(tmp_path):
    """The seed fixes the train/val split and the event order, so a pool restored under a different
    one would be indexed against a different stream."""
    path = str(tmp_path / "trainer.npz")
    _, original = _trainer(DesignTrainer, seed=3)
    original.persist(path)
    io.commit([path])
    _, other = _trainer(DesignTrainer, seed=4)
    with pytest.raises(ValueError, match="was written at seed"):
        other.restore(path)


def test_continual_trainer_restores_its_persistent_network(tmp_path):
    """The continual strategy carries ONE network across designs, so resuming without it would throw
    away every design's training. Params and buffer state come back exactly, and they are ALL that
    crosses a design boundary -- the optimiser is restarted at each one either way -- so a resumed
    continual run starts the next design from exactly the state an uninterrupted one would."""
    path = str(tmp_path / "trainer.npz")
    detector, original = _trainer(ContinualTrainer, seed=5)
    design = np.full(detector.design_dim(), 0.5, np.float32)
    assert original.train(design, 21) is not None
    original.persist(path)
    io.commit([path])

    _, restored = _trainer(ContinualTrainer, seed=5)
    before = jax.tree.leaves(restored._running[0])
    restored.restore(path)
    trained, after = jax.tree.leaves(original._running[0]), jax.tree.leaves(restored._running[0])
    assert any(not np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(before, trained))
    for x, y in zip(trained, after):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_a_resumed_run_warm_starts_from_a_design_it_did_not_train(tmp_path):
    """The property the driver's warm start rests on ACROSS a resume, tested on the pipeline.

    ``scripts/bo.py`` holds no historical parameters. ``continue`` and ``closest`` choose a design
    NUMBER out of ``proposed_scaled`` -- which the resume rebuilds from ``partial.json``, so it spans
    the whole run -- and the trainer reads that design's checkpoint. A process that trained NONE of
    those designs must therefore recover any of them, which is what this asserts: a second trainer,
    holding nothing, reproduces the first one's trained parameters exactly.

    It is the case the previous scheme could not serve. The driver kept the networks in a list that a
    resume left empty while restoring the design rows beside it, so ``closest`` indexed a one-element
    list with a row number drawn from the full history and went out of range on the second design after
    any restart."""
    detector, original = _trainer(DesignTrainer, seed=3, checkpoint_dir=str(tmp_path / "checkpoints"))
    design = np.full(detector.design_dim(), 0.25, np.float32)
    result = original.train(design, 11, step=0)
    assert result is not None

    _, resumed = _trainer(DesignTrainer, seed=3, checkpoint_dir=str(tmp_path / "checkpoints"))
    warm = jax.tree.leaves(resumed.restore_design_parameters(0))
    trained = jax.tree.leaves(result.params)
    assert len(trained) > 0 and len(trained) == len(warm)
    for x, y in zip(trained, warm):
        assert np.array_equal(np.asarray(x), np.asarray(y))


# ---------------------------------------------------------------------- #
# The seed sequence
# ---------------------------------------------------------------------- #
def test_replaying_the_seed_branch_lands_on_the_same_stream():
    """Resume is 'spawn k and carry on'. If that did not reproduce the original stream, a resumed run
    would silently be a different run from the design it restarts on."""
    def stream(skip, count):
        _, iteration = np.random.SeedSequence(1234).spawn(2)
        iteration.spawn(skip)
        return [int(iteration.spawn(1)[0].generate_state(1)[0]) for _ in range(count)]

    assert stream(0, 6)[3:] == stream(3, 3)


def test_bayes_risk_is_the_floor_the_task_is_scored_against():
    """The debug task's answer, stated so a driver can be checked against it rather than against its
    own output: the corners are the optimum and coincident probes are worthless."""
    detector = LinearDetector()
    corners = detector.bayes_risk(np.array([-1.0, 1.0], np.float32))
    assert corners == pytest.approx(detector.bayes_risk(np.array([1.0, -1.0], np.float32)))  # a SET
    assert corners < 0.01 < 0.4 < detector.bayes_risk(np.array([1.0, 1.0], np.float32))
    # ... and it is a genuine optimum over the box, not merely better than one bad point.
    rng = np.random.default_rng(0)
    sample = rng.uniform(-1.0, 1.0, size=(2000, 2)).astype(np.float32)
    assert min(detector.bayes_risk(x) for x in sample) > corners
