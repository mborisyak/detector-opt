"""A gradient-boosted-tree stand-in for the per-design regressor.

The BO objective is "how well can a regressor trained on this design's readout recover the target".
Nothing about that requires a neural network -- it requires an estimator that saturates the design's
information content and reports where it saturated. On the enzyme benchmark a GBDT does that in
**~0.7 s** against the meta-regressor's 27-137 s, which is what makes a 100+ iteration BO study
affordable at all.

The recipe, deliberately budget-free: sample generously at the design, fit ONE GBDT with many more
base learners than needed, take the validation minimum -- the overfitting point, where adding
learners stops buying generalisation -- and report ``(train + val) / 2`` there with its standard
error.

The estimator is **XGBoost** (user, 2026-08-11), previously sklearn's ``HistGradientBoostingRegressor``.
The recipe is unchanged; only how the stopping point is found differs. XGBoost reports the whole
validation curve in ``evals_result`` and remembers its minimum as ``best_iteration``, so the score is
read with ONE prediction at ``iteration_range=(0, best + 1)`` instead of walking every stage --
cheaper, and it keeps the per-SAMPLE errors the SEM needs. Samples are then thrown away; no pool, no
warm start, no shared budget. That matches the convention the neural objective already uses
(``scripts/bo.py``: the objective is ``(mean_train + mean_val) / 2`` and the GP noise is its SEM), so
the two are directly comparable.

Averaging train with val is not a mistake: val alone, minimised over stages, is biased low by the
selection, and train alone is biased low by fitting. The average of the two at the stopping point is
the stable quantity, and it is what the neural pipeline reports as well.

THREE TARGET SHAPES, one per candidate task, because a proxy that reported a different quantity from
the detector's own ``loss`` would be screening a different problem:

* **scalar** (``enzyme`` melting point, ``growth`` optimal temperature) -- one regressor, squared
  error. This path is unchanged from before the other two existed, so the numbers already measured
  for those candidates reproduce exactly.
* **vector** (``enzyme_mm`` recovers three Michaelis-Menten coefficients at once) -- one regressor
  PER COMPONENT, each stopped at its OWN validation minimum, loss = the mean over components. They
  saturate at different numbers of learners, so a shared stopping point would under-fit one and
  over-fit another; the mean over components is exactly what ``detector.loss`` computes.
* **one-hot class** (``enzyme_inhib``: which of three inhibition mechanisms) -- an XGBoost CLASSIFIER
  (``multi:softprob``, early-stopped on ``mlogloss``) reporting per-sample cross-entropy divided by
  ``ln K``, again the same quantity ``detector.loss`` computes. A detector selects this path by
  carrying an ``n_classes`` attribute; nothing is inferred from the target's width, because a width-3
  one-hot and a 3-component regression target are indistinguishable by shape alone.
"""

import os
from typing import NamedTuple

import numpy as np

import xgboost as xgb

__all__ = ["GBDTScore", "score_design", "sample_design"]


class GBDTScore(NamedTuple):
  """One design's score. ``loss`` is what BO minimises, ``sem`` is the GP's observation noise.

  ``train``/``val`` are MSE for a scalar or vector target and cross-entropy / ``ln K`` for a class.

  ``train_sem``/``val_sem`` are the two sides' OWN standard errors, reported separately because the
  neural convergence procedure (``detopt/nn/trainer/design.py``) spells its resolution as
  ``|val - train| + hypot(train_sem, val_sem)`` and ``sem`` -- the standard error of the AVERAGE --
  is not that quantity. Both are the per-SAMPLE spread over the events at the stopping point."""
  loss: float  # (train + val) / 2 at the validation minimum
  sem: float  # its standard error
  train: float  # train loss at the stopping point
  val: float  # val loss at the stopping point (the minimum over stages)
  n_learners: int  # the stopping point itself: how many base learners the design could support
  n_events: int
  train_sem: float = float("nan")
  val_sem: float = float("nan")


def n_classes_of(detector) -> int:
  """``K`` if this detector's target is a one-hot class label, else 0 (a scalar/vector regression).

  Read off an ATTRIBUTE rather than guessed from the target's shape: a one-hot 3-vector and a
  3-component regression target are the same array shape and must not be scored the same way.
  """
  return int(getattr(detector, "n_classes", 0))


def sample_design(detector, design, event_index):
  """``(features, target)`` for one design: every event's flattened readout ``(n_events, n_features)``
  and its normalised target ``(n_events, n_target)``.

  The design columns of ``combine_scaled`` are deliberately NOT included -- within one design they
  are constant, so they carry nothing for a per-design estimator. The readout is flattened rather
  than treated as a set: for a FIXED design the experiments are distinguishable (each has its own
  temperature and enzyme fraction), so the permutation invariance the deep set buys across designs
  is not a symmetry here, and column ``i`` means the same thing in every row.

  The target keeps its component axis: a VECTOR target is ``n_target`` separate regressions and a
  one-hot is a class, so ravelling would silently interleave the components of the first and destroy
  the second.
  """
  _, event, _, target = detector(design, np.asarray(event_index, np.int64))
  n_events = len(event_index)
  features = np.asarray(event.measurements, np.float32).reshape(n_events, -1)
  return features, np.asarray(detector.normalize_target(target), np.float32).reshape(n_events, -1)


def score_design(
  detector, design, *, n_events: int, event_offset: int = 0, val_fraction: float = 0.25, max_learners: int = 400,
  learning_rate: float = 0.08, max_leaf_nodes: int = 31, min_samples_leaf: int = 40, patience: int = 50,
  n_threads: int | None = None, seed: int = 0
) -> GBDTScore:
  """Score one design (see the module docstring). Lower is better; the no-information level is the
  target's own variance (1/3 for the uniform prior the enzyme benchmark normalises to) for a
  regression target, and exactly 1.0 for a class target scored in units of ``ln K``.

  ``event_offset`` picks WHICH events are drawn. The detector seeds an event from its index alone, so
  a fixed offset re-measures the same enzymes under every design -- common random numbers, which
  removes the draw from the design-to-design comparison. Passing a per-design offset instead makes
  every evaluation an independent draw.
  """
  event_index = np.arange(event_offset, event_offset + int(n_events), dtype=np.int64)
  features, target = sample_design(detector, design, event_index)
  n_classes = n_classes_of(detector)

  n_train = int(round((1.0 - val_fraction) * len(target)))
  train_features, val_features = features[:n_train], features[n_train:]
  train_target, val_target = target[:n_train], target[n_train:]

  settings = dict(
    max_learners=max_learners, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
    patience=patience, n_threads=n_threads, seed=seed
  )

  if n_classes > 1:
    # ONE classifier over all K classes (not K regressions): the classes are mutually exclusive and
    # softmax couples them, which is the same coupling the neural objective's cross-entropy has.
    train_loss, val_loss, stopping = _fit_class(
      train_features, np.argmax(train_target, axis=1), val_features, np.argmax(val_target, axis=1), n_classes=n_classes,
      **settings
    )
    stopping = [stopping]
  else:
    # ONE GBDT PER TARGET COMPONENT, each stopped at its OWN validation minimum. For a SCALAR target
    # the loop runs once and reduces bit-identically to the original recipe.
    train_squared = np.empty_like(train_target)
    val_squared = np.empty_like(val_target)
    stopping = []
    for component in range(target.shape[1]):
      train_squared[:, component], val_squared[:, component], stopped = _fit_component(
        train_features, train_target[:, component], val_features, val_target[:, component], **settings
      )
      stopping.append(stopped)
    # Per-SAMPLE loss = the mean over components, i.e. what `detector.loss` returns per event.
    train_loss, val_loss = train_squared.mean(axis=1), val_squared.mean(axis=1)

  # The SEM comes from the per-SAMPLE spread, not from an average of averages.
  train_mean, val_mean = float(train_loss.mean()), float(val_loss.mean())
  # loss = (train + val) / 2, so its variance is a quarter of the sum of the two means' variances.
  sem = 0.5 * float(np.sqrt(train_loss.var() / train_loss.size + val_loss.var() / val_loss.size))
  train_sem = float(np.sqrt(train_loss.var() / train_loss.size))
  val_sem = float(np.sqrt(val_loss.var() / val_loss.size))
  return GBDTScore(
    loss=0.5 * (train_mean + val_mean),
    sem=sem,
    train=train_mean,
    val=val_mean,
    # the DEEPEST component's stopping point: how many base learners the design could support at all
    n_learners=int(max(stopping)) + 1,
    n_events=int(n_events),
    train_sem=train_sem,
    val_sem=val_sem
  )


def _parameters(*, learning_rate, max_leaf_nodes, min_samples_leaf, n_threads, seed):
  """The settings shared by every fit here. `lossguide` + `max_leaves` reproduces the leaf-wise growth
  HistGradientBoostingRegressor used, and `min_child_weight` is the leaf-size floor (for squared error
  the child weight IS the sample count)."""
  return {
    'eta': float(learning_rate),
    'tree_method': 'hist',
    'grow_policy': 'lossguide',
    'max_leaves': int(max_leaf_nodes),
    'max_depth': 0,  # unbounded depth: `max_leaves` is the capacity knob under lossguide
    'min_child_weight': int(min_samples_leaf),
    'seed': int(seed),
    # THREADS: default to the CPUs this job was ALLOCATED, not to every core on the machine.
    # XGBoost's own default (nthread=0) takes all visible cores, and SLURM's cgroup restricts WHICH
    # cores a job may use but not HOW MANY threads it spawns -- so several jobs each start ~12
    # threads, pile them onto 4 allocated cores, and the machine runs at load 44 while doing the
    # work of about six. Measured: three concurrent screens ran 10-14x slower than the same screen
    # alone. `SLURM_CPUS_PER_TASK` is what the scheduler promised us; outside SLURM fall back to a
    # modest 4 rather than the whole box.
    'nthread': int(n_threads) if n_threads is not None else int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
  }


def _fit_component(
  train_features, train_target, val_features, val_target, *, max_learners, learning_rate, max_leaf_nodes, min_samples_leaf,
  patience, n_threads, seed
):
  """One component of a regression target: fit, stop at the validation minimum, return the per-SAMPLE
  squared errors there (train, val) and the stopping index."""
  train_matrix = xgb.DMatrix(train_features, label=train_target)
  val_matrix = xgb.DMatrix(val_features, label=val_target)
  parameters = _parameters(
    learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, n_threads=n_threads,
    seed=seed
  )
  parameters['objective'] = 'reg:squarederror'
  # `early_stopping_rounds` finds the validation minimum without fitting all `max_learners` trees;
  # the patience is generous so a plateau is not mistaken for the minimum.
  model = xgb.train(
    parameters, train_matrix, num_boost_round=int(max_learners), evals=[(val_matrix, 'val')],
    early_stopping_rounds=int(patience), verbose_eval=False
  )
  stopping = int(model.best_iteration)

  # Per-SAMPLE squared errors AT THE STOPPING POINT.
  stage = (0, stopping + 1)
  train_squared = np.square(model.predict(train_matrix, iteration_range=stage) - train_target)
  val_squared = np.square(model.predict(val_matrix, iteration_range=stage) - val_target)
  return train_squared, val_squared, stopping


def _fit_class(
  train_features, train_label, val_features, val_label, *, n_classes, max_learners, learning_rate, max_leaf_nodes,
  min_samples_leaf, patience, n_threads, seed
):
  """A one-hot class target: fit a softmax classifier, stop at the validation minimum of ``mlogloss``
  (the SAME quantity the score reports, exactly as the regressor stops on validation MSE), and return
  the per-SAMPLE cross-entropies there in units of ``ln K`` -- so a uniform prediction scores exactly
  1.0 on EVERY sample. That is the point of the normalisation: an uninformative design's per-sample
  loss has ~zero variance, so its standard error collapses at once instead of making it the
  expensive one to evaluate."""
  train_matrix = xgb.DMatrix(train_features, label=train_label)
  val_matrix = xgb.DMatrix(val_features, label=val_label)
  parameters = _parameters(
    learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf, n_threads=n_threads,
    seed=seed
  )
  parameters.update({'objective': 'multi:softprob', 'num_class': int(n_classes), 'eval_metric': 'mlogloss'})
  model = xgb.train(
    parameters, train_matrix, num_boost_round=int(max_learners), evals=[(val_matrix, 'val')],
    early_stopping_rounds=int(patience), verbose_eval=False
  )
  stopping = int(model.best_iteration)

  stage = (0, stopping + 1)
  scale = float(np.log(n_classes))

  def cross_entropy(probabilities, labels):
    picked = probabilities[np.arange(len(labels)), labels.astype(np.int64)]
    return -np.log(np.clip(picked, 1e-12, None)) / scale

  train_errors = cross_entropy(model.predict(train_matrix, iteration_range=stage), train_label)
  val_errors = cross_entropy(model.predict(val_matrix, iteration_range=stage), val_label)
  return train_errors, val_errors, stopping
