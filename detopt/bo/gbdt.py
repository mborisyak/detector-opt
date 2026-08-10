"""A gradient-boosted-tree stand-in for the per-design regressor.

The BO objective is "how well can a regressor trained on this design's readout recover the target".
Nothing about that requires a neural network -- it requires an estimator that saturates the design's
information content and reports where it saturated. On the enzyme benchmark a GBDT does that in
**~0.7 s** against the meta-regressor's 27-137 s, which is what makes a 100+ iteration BO study
affordable at all.

The recipe, deliberately budget-free: sample generously at the design, fit ONE GBDT with many more
base learners than needed, walk its ``staged_predict`` to the validation minimum -- the overfitting
point, where adding learners stops buying generalisation -- and report ``(train + val) / 2`` there
with its standard error. Samples are then thrown away; no pool, no warm start, no shared budget.
That matches the convention the neural objective already uses (``scripts/bo.py``: the objective is
``(mean_train + mean_val) / 2`` and the GP noise is its SEM), so the two are directly comparable.

Averaging train with val is not a mistake: val alone, minimised over stages, is biased low by the
selection, and train alone is biased low by fitting. The average of the two at the stopping point is
the stable quantity, and it is what the neural pipeline reports as well.
"""

from typing import NamedTuple

import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor

__all__ = ["GBDTScore", "score_design", "sample_design"]


class GBDTScore(NamedTuple):
  """One design's score. ``loss`` is what BO minimises, ``sem`` is the GP's observation noise."""
  loss: float  # (train + val) / 2 at the validation minimum
  sem: float  # its standard error
  train: float  # train MSE at the stopping point
  val: float  # val MSE at the stopping point (the minimum over stages)
  n_learners: int  # the stopping point itself: how many base learners the design could support
  n_events: int


def sample_design(detector, design, event_index):
  """``(features, target)`` for one design: every event's flattened readout and its normalised target.

  The design columns of ``combine_scaled`` are deliberately NOT included -- within one design they
  are constant, so they carry nothing for a per-design estimator. The readout is flattened rather
  than treated as a set: for a FIXED design the experiments are distinguishable (each has its own
  temperature and enzyme fraction), so the permutation invariance the deep set buys across designs
  is not a symmetry here, and column ``i`` means the same thing in every row.
  """
  _, event, _, target = detector(design, np.asarray(event_index, np.int64))
  features = np.asarray(event.measurements, np.float32).reshape(len(event_index), -1)
  return features, np.asarray(detector.normalize_target(target), np.float32).ravel()


def score_design(
  detector, design, *, n_events: int, event_offset: int = 0, val_fraction: float = 0.25, max_learners: int = 400,
  learning_rate: float = 0.08, max_leaf_nodes: int = 31, min_samples_leaf: int = 40, seed: int = 0
) -> GBDTScore:
  """Score one design (see the module docstring). Lower is better; the no-information level is the
  target's own variance (1/3 for the uniform prior the enzyme benchmark normalises to).

  ``event_offset`` picks WHICH events are drawn. The detector seeds an event from its index alone, so
  a fixed offset re-measures the same enzymes under every design -- common random numbers, which
  removes the draw from the design-to-design comparison. Passing a per-design offset instead makes
  every evaluation an independent draw.
  """
  event_index = np.arange(event_offset, event_offset + int(n_events), dtype=np.int64)
  features, target = sample_design(detector, design, event_index)

  n_train = int(round((1.0 - val_fraction) * len(target)))
  train_features, val_features = features[:n_train], features[n_train:]
  train_target, val_target = target[:n_train], target[n_train:]

  model = HistGradientBoostingRegressor(
    max_iter=int(max_learners),
    learning_rate=float(learning_rate),
    max_leaf_nodes=int(max_leaf_nodes),
    min_samples_leaf=int(min_samples_leaf),
    early_stopping=False,  # the staged walk below IS the early stopping, and it needs the full curve
    random_state=int(seed)
  )
  model.fit(train_features, train_target)

  # Squared errors per stage, kept per SAMPLE at the chosen stage so the SEM is the data's own.
  val_squared = [np.square(prediction - val_target) for prediction in model.staged_predict(val_features)]
  stopping = int(np.argmin([squared.mean() for squared in val_squared]))
  train_squared = None
  for stage, prediction in enumerate(model.staged_predict(train_features)):
    if stage == stopping:
      train_squared = np.square(prediction - train_target)
      break

  train_mse, val_mse = float(train_squared.mean()), float(val_squared[stopping].mean())
  # loss = (train + val) / 2, so its variance is a quarter of the sum of the two means' variances.
  sem = 0.5 * float(
    np.sqrt(train_squared.var() / train_squared.size + val_squared[stopping].var() / val_squared[stopping].size)
  )
  return GBDTScore(
    loss=0.5 * (train_mse + val_mse), sem=sem, train=train_mse, val=val_mse, n_learners=stopping + 1, n_events=int(n_events)
  )
