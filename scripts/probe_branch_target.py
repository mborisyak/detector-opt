#!/usr/bin/env python3
"""Is the inhibitor task a better BENCHMARK as REGRESSION on (log10 k1, log10 k2) than as a 3-way class?

    python scripts/probe_branch_target.py =enzyme_inhib --results output/screen/sobol-designs-m4.json \
        --rank 999 --n-events 209715 --output output/screen/branch-good.json

WHY. As a CLASSIFICATION problem the task is too hard to leave BO much room. Chance is exactly 1.0
(cross-entropy / ln 3), and the best measured anywhere -- XGBoost, 209 715 events, at a good design --
is 0.652, i.e. mean -ln p(correct) = 0.714 nats, about 0.49 probability on the true class against 0.33
for a coin. The whole usable range is therefore ~0.35 for a PERFECT estimator, and the deep set
actually reaches 0.756, leaving BO a span of ~0.19. A narrow span is what makes criterion (d)'s bar
(`10 x loss_precision`) unreachable at any tolerable multiple.

The class label is a THRESHOLDED function of `d = log10(k1/k2)`: the detector draws the two branch
affinities and then reports which band `|d|` fell in. Thresholding is where the information goes --
a compound at |d| = 2.4 and one at |d| = 1.1 are the same label. Regressing the two branches keeps it.

WHAT THIS MEASURES, and what it does NOT. It needs no change to the detector: the read-outs are
unchanged (`combine_scaled`, exactly what the network sees), and `log10_k1` / `log10_k2` are already
carried in the GROUND TRUTH. So this is the SAME experiment scored against a different target, which
is precisely the comparison wanted. It is a SCREEN, not the task: a real regression variant would also
drop the structural gap in the preference prior (0.15 < |d| < 1.0 is not sampled, which exists only to
keep the CLASSES unambiguous) and would draw `d` continuously. That gap makes the target here BIMODAL,
which inflates its variance and hence the no-information level -- so the span reported below is an
optimistic reading of a variant that has not been built. Stated here so the number is not over-read.

NORMALISATION. Both branches are mapped onto [-1, 1] by the detector's own `log10_branch_bounds`, and
the reported loss is the mean squared error over the two components -- the convention every other
candidate uses, so the no-information level is the target's own variance (printed alongside).
"""

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _threads)

import numpy as np
import xgboost as xgb
import yaml

import detopt
import detopt.detector


def build(detector, design_scaled, n_events, chunk=16384):
  """`(features (n, m*F), branches (n, 2), class_label (n,))` at ONE design.

  `detector.__call__` takes the NOMINAL design (physical units); `combine_scaled` takes the SCALED
  one. Confusing the two silently simulates a nonsense experiment -- see SLOP-REPORT.md.
  """
  design_scaled = np.asarray(design_scaled, np.float32)
  design_nominal = np.asarray(detector.flatten_design(detector.to_nominal(design_scaled)), np.float32)
  features, branches, labels = [], [], []
  done = 0
  while done < n_events:
    size = min(chunk, n_events - done)
    index = np.arange(done, done + size, dtype=np.int64)
    ground_truth, event, _, target = detector(design_nominal, index)
    features.append(np.asarray(detector.combine_scaled(event, design_scaled), np.float32).reshape(size, -1))
    branches.append(
      np.stack([
        np.asarray(ground_truth.log10_k1, np.float32).reshape(size),
        np.asarray(ground_truth.log10_k2, np.float32).reshape(size)
      ], axis=1)
    )
    labels.append(np.argmax(np.asarray(detector.normalize_target(target), np.float32), axis=1))
    done += size
  return np.concatenate(features), np.concatenate(branches), np.concatenate(labels).astype(np.int64)


def fit_component(train_features, train_target, val_features, val_target, max_learners, seed, threads):
  """One regression component; returns per-sample squared errors at the VALIDATION MINIMUM."""
  train_matrix = xgb.DMatrix(train_features, label=train_target)
  val_matrix = xgb.DMatrix(val_features, label=val_target)
  history = {}
  model = xgb.train(
    {
      'objective': 'reg:squarederror', 'eta': 0.08, 'tree_method': 'hist', 'grow_policy': 'lossguide',
      'max_leaves': 31, 'max_depth': 0, 'min_child_weight': 40, 'seed': seed, 'nthread': threads,
    },
    train_matrix, num_boost_round=max_learners,
    evals=[(train_matrix, 'train'), (val_matrix, 'val')], evals_result=history, verbose_eval=False
  )
  stopping = int(np.argmin(history['val']['rmse']))
  stage = (0, stopping + 1)
  train_squared = np.square(model.predict(train_matrix, iteration_range=stage) - train_target)
  val_squared = np.square(model.predict(val_matrix, iteration_range=stage) - val_target)
  return train_squared, val_squared, stopping


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config")
  parser.add_argument("--results", required=True)
  parser.add_argument("--rank", type=int, default=999, help="999 = the BEST design in the file")
  parser.add_argument("--target", choices=("branch", "extremes"), default="branch",
                      help="branch = regress (log10 k1, log10 k2); extremes = BINARY competitive vs "
                           "UNcompetitive, i.e. sign(d) among |d| >= 1 only, scored as cross-entropy / ln 2. "
                           "The extremes are the pair the 2x2 factorial exists to separate: competitive is "
                           "visible only BELOW K_B and uncompetitive only at SATURATION, so a design with a "
                           "single ATP level cannot tell them apart at all -- maximal design dependence.")
  parser.add_argument("--n-events", type=int, default=209715)
  parser.add_argument("--val-fraction", type=float, default=0.25)
  parser.add_argument("--max-learners", type=int, default=2000)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", default="output/screen/branch.json")
  arguments = parser.parse_args()

  name = arguments.config.lstrip("=")
  with open(f"config/{name}.yaml") as f:
    config = yaml.safe_load(f)
  detector_config = config["detector"]
  if isinstance(detector_config, str):
    with open(f"config/detector/{detector_config}.yaml") as f:
      detector_config = yaml.safe_load(f)
  detector = detopt.detector.from_config(detector_config)

  with open(arguments.results) as f:
    recorded = json.load(f)["results"]
  order = sorted(recorded, key=lambda r: -r["loss"])
  chosen = order[min(arguments.rank, len(order) - 1)]
  print(f"design rank {arguments.rank}: proxy CLASS loss {chosen['loss']:.4f}")

  features, branches, labels = build(detector, np.asarray(chosen["x_scaled"], np.float32), arguments.n_events)

  if arguments.target == "extremes":
    # BINARY competitive vs UNcompetitive: sign(d) among the well-separated compounds only.
    # `d = log10(k1/k2)` > 0 means the compound prefers the FREE enzyme (competitive), < 0 the ES
    # complex (uncompetitive). Keeping |d| >= 1 drops the noncompetitive band, so the two remaining
    # classes are the pair the 2x2 factorial exists to separate -- competitive shows only BELOW K_B,
    # uncompetitive only at SATURATION, so a single-ATP design cannot tell them apart at all.
    difference = branches[:, 0] - branches[:, 1]
    keep = np.abs(difference) >= 1.0
    binary = (difference[keep] > 0).astype(np.int64)
    kept_features = features[keep]
    scale = float(np.log(2.0))
    print(f"extremes: kept {keep.sum()} of {len(keep)} events (|d| >= 1); "
          f"competitive {binary.mean():.3f} / uncompetitive {1 - binary.mean():.3f}")
    n_train = int(round((1.0 - arguments.val_fraction) * len(binary)))
    threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    train_matrix = xgb.DMatrix(kept_features[:n_train], label=binary[:n_train])
    val_matrix = xgb.DMatrix(kept_features[n_train:], label=binary[n_train:])
    history = {}
    model = xgb.train(
      {
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'eta': 0.08, 'tree_method': 'hist',
        'grow_policy': 'lossguide', 'max_leaves': 31, 'max_depth': 0, 'min_child_weight': 40,
        'seed': arguments.seed, 'nthread': threads,
      },
      train_matrix, num_boost_round=arguments.max_learners,
      evals=[(train_matrix, 'train'), (val_matrix, 'val')], evals_result=history, verbose_eval=False
    )
    stopping = int(np.argmin(history['val']['logloss']))
    stage = (0, stopping + 1)

    def cross_entropy(probability, y):
      picked = np.where(y == 1, probability, 1.0 - probability)
      return -np.log(np.clip(picked, 1e-12, None)) / scale

    train_errors = cross_entropy(model.predict(train_matrix, iteration_range=stage), binary[:n_train])
    val_errors = cross_entropy(model.predict(val_matrix, iteration_range=stage), binary[n_train:])
    guess = (model.predict(val_matrix, iteration_range=stage) > 0.5).astype(np.int64)
    accuracy = float((guess == binary[n_train:]).mean())
    train_mean, val_mean = float(train_errors.mean()), float(val_errors.mean())
    level = 0.5 * (train_mean + val_mean)
    err = 0.5 * float(np.sqrt(train_errors.var() / train_errors.size + val_errors.var() / val_errors.size))
    print(f"\n  stopped at {stopping + 1} learners | train {train_mean:.5f} val {val_mean:.5f} "
          f"level {level:.5f} diff {abs(train_mean - val_mean):.5f} err {err:.5f}")
    print(f"  ACCURACY {accuracy:.4f} (chance 0.5) | no-information level is EXACTLY 1.0 (CE / ln 2)")
    print(f"  span at this design: 1.0000 -> {level:.4f} = {1 - level:.4f} ({100 * (1 - level):.0f}% down)")
    with open(arguments.output, "w") as f:
      json.dump({
        "design": chosen, "target": "extremes", "n_events": arguments.n_events, "kept": int(keep.sum()),
        "ceiling": 1.0, "level": level, "train": train_mean, "val": val_mean,
        "diff": abs(train_mean - val_mean), "err": err, "accuracy": accuracy, "stopping": stopping + 1,
      }, f, indent=2, default=float)
    print(f"\nwrote {arguments.output}")
    return

  # Normalise both branches onto [-1, 1] with the DETECTOR's own bounds -- the same convention every
  # other candidate uses, so the no-information level is the target's own variance.
  low, high = detector.log10_branch_bounds
  centre, half = 0.5 * (high + low), 0.5 * (high - low)
  normalised = (branches - centre) / half
  ceiling = float(normalised.var(axis=0).mean())
  print(f"branch bounds {low:.3f}..{high:.3f} | normalised target variance (NO-INFORMATION level) {ceiling:.4f}")
  print(f"  per component: {np.array2string(normalised.var(axis=0), precision=4)}")

  n_train = int(round((1.0 - arguments.val_fraction) * len(labels)))
  threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
  train_squared, val_squared, stopping = [], [], []
  for component in range(2):
    tr, va, stop = fit_component(
      features[:n_train], normalised[:n_train, component], features[n_train:], normalised[n_train:, component],
      arguments.max_learners, arguments.seed, threads
    )
    train_squared.append(tr)
    val_squared.append(va)
    stopping.append(stop + 1)

  train_loss = np.mean(np.stack(train_squared, axis=1), axis=1)
  val_loss = np.mean(np.stack(val_squared, axis=1), axis=1)
  train_mean, val_mean = float(train_loss.mean()), float(val_loss.mean())
  level = 0.5 * (train_mean + val_mean)
  diff = abs(train_mean - val_mean)
  err = 0.5 * float(np.sqrt(train_loss.var() / train_loss.size + val_loss.var() / val_loss.size))

  print(f"\n  stopped at {stopping} learners (per component, validation minimum)")
  print(f"  train {train_mean:.5f}  val {val_mean:.5f}  level {level:.5f}  diff {diff:.5f}  err {err:.5f}")
  print(f"\n  REGRESSION span at this design: {ceiling:.4f} (chance) -> {level:.4f}"
        f"  = {ceiling - level:.4f}, i.e. {100 * (1 - level / ceiling):.0f}% of the way down")
  print(f"  CLASSIFICATION span for contrast: 1.0000 (chance) -> 0.652 measured = 0.348 "
        f"({100 * (1 - 0.652):.0f}%), and the deep set only reaches 0.756")

  with open(arguments.output, "w") as f:
    json.dump({
      "design": chosen, "n_events": arguments.n_events, "ceiling": ceiling, "level": level,
      "train": train_mean, "val": val_mean, "diff": diff, "err": err, "stopping": stopping,
      "component_variance": normalised.var(axis=0).tolist(),
    }, f, indent=2, default=float)
  print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
  main()
