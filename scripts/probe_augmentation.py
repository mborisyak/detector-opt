#!/usr/bin/env python3
"""How small can the train/validation GAP be at one design, with a well-regularised estimator?

    python scripts/probe_augmentation.py =enzyme_inhib \
        --results output/campaign-inhib/126382657/from_scratch/results.json \
        --permutations 10 --output output/screen/augment-p10.json
    #  ... and the CONTROL, same everything, no augmentation:
    python scripts/probe_augmentation.py =enzyme_inhib --results ... --permutations 0 ...

WHY. The neural campaign cannot converge on uninformative designs: the trainer stops when
`diff + err < loss_precision`, `err` is the loss estimate's standard error (falls like `1/sqrt(n)`)
and `diff` is the train/validation gap, which is an OVERFITTING BIAS and does not fall with more data.
Measured, `diff` alone reaches 0.0083, which forced `loss_precision` to 8.0e-3 and put criterion (d)'s
bar (`10 x loss_precision`) at 0.080 against an achievable span of ~0.09-0.20.

The question this answers is whether that floor belongs to the TASK or to the ESTIMATOR. It takes a
tenth of the campaign's per-run budget at ONE design and fits XGBoost instead of the deep set, with
the stopping rule stated by the user: **stop at the number of learners where `|val - train|` is
MINIMAL** (not at the validation minimum, which is where `detopt.bo.gbdt` stops -- that one maximises
accuracy, this one minimises the very gap that blocks convergence). Then it reports the SAME error
definition the trainer uses, `diff + err`, so the numbers are directly comparable to the 0.0083.

THE AUGMENTATION. The experiments in a batch are a SET: the target is invariant under permuting them,
PROVIDED each experiment's own design coordinates travel with its read-outs. `combine_scaled` returns
exactly that pairing -- `(n_experiments, n_measurements + 4)`, each row a experiment's [A] samples
followed by its own four design values -- so permuting along the experiment axis is an exact symmetry
of the label and a valid augmentation. `--permutations 0` runs the identical pipeline WITHOUT it, so
the augmentation's effect is isolated rather than confounded with the estimator change.

LEAKAGE. Permutations of one event are near-duplicates, so the split is done over EVENTS FIRST and
every permutation of an event lands on the same side. Splitting after augmenting would put copies of
the same event in train and validation and drive the gap to zero for the wrong reason.
"""

import argparse
import json
import os


# Cap BLAS threads BEFORE numpy is imported (SLURM restricts WHICH cores, not how many threads).
_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
  os.environ.setdefault(_v, _threads)

import numpy as np
import xgboost as xgb
import yaml

import detopt
import detopt.detector


def build(detector, design_scaled, n_events, chunk=16384):
  """`(features (n, m, F), label (n,))` at ONE design, sampled in chunks to bound memory.

  TWO DESIGN SPACES, and they must not be confused. ``detector.__call__`` simulates a NOMINAL
  (physical) design -- degrees Celsius, mM -- while ``combine_scaled`` wants the SCALED [0, 1] vector,
  because that is what the network is conditioned on. Passing the scaled vector to ``__call__`` runs a
  nonsense experiment (temperature ~0.5 C, concentrations ~0.5 mM) that has nothing to do with the
  design being studied, and it fails SILENTLY -- the numbers look plausible and are simply wrong.
  """
  design_scaled = np.asarray(design_scaled, np.float32)
  design_nominal = np.asarray(detector.flatten_design(detector.to_nominal(design_scaled)), np.float32)
  features, labels = [], []
  done = 0
  while done < n_events:
    size = min(chunk, n_events - done)
    index = np.arange(done, done + size, dtype=np.int64)
    _, event, _, target = detector(design_nominal, index)
    combined = np.asarray(detector.combine_scaled(event, design_scaled), np.float32)
    features.append(combined)
    labels.append(np.argmax(np.asarray(detector.normalize_target(target), np.float32), axis=1))
    done += size
  return np.concatenate(features), np.concatenate(labels).astype(np.int64)


def augment(features, labels, n_permutations, rng):
  """Permute the EXPERIMENT axis. Each experiment's design coordinates sit in its own row, so a
  permutation leaves the label untouched -- an exact symmetry, not a heuristic distortion."""
  if n_permutations <= 0:
    return features.reshape(len(features), -1), labels
  n, m, _ = features.shape
  out_features, out_labels = [], []
  for _ in range(n_permutations):
    order = np.argsort(rng.random((n, m)), axis=1)  # an independent permutation per event
    out_features.append(np.take_along_axis(features, order[:, :, None], axis=1).reshape(n, -1))
    out_labels.append(labels)
  return np.concatenate(out_features), np.concatenate(out_labels)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("config")
  parser.add_argument("--results", required=True)
  parser.add_argument("--rank", type=int, default=0, help="0 = the WORST design the search visited")
  parser.add_argument("--permutations", type=int, default=10, help="0 = the control, no augmentation")
  parser.add_argument("--n-events", type=int, default=209715, help="a tenth of the campaign budget 2097152")
  parser.add_argument("--val-fraction", type=float, default=0.25)
  parser.add_argument("--max-learners", type=int, default=600)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", default="output/screen/augment.json")
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
  design_scaled = np.asarray(chosen["x_scaled"], np.float32)
  print(f"design rank {arguments.rank}: recorded neural loss {chosen['loss']:.4f}"
        f"  (worst {order[0]['loss']:.4f}, best {order[-1]['loss']:.4f})")

  features, labels = build(detector, design_scaled, arguments.n_events)
  n_classes = int(labels.max()) + 1
  scale = float(np.log(n_classes))

  # SPLIT OVER EVENTS FIRST, then augment each side separately -- permutations of one event are
  # near-duplicates and must not straddle the split.
  n_train = int(round((1.0 - arguments.val_fraction) * len(labels)))
  rng = np.random.default_rng(arguments.seed)
  train_features, train_labels = augment(features[:n_train], labels[:n_train], arguments.permutations, rng)
  val_features, val_labels = augment(features[n_train:], labels[n_train:], arguments.permutations, rng)
  print(f"rows: train {train_features.shape}, val {val_features.shape}"
        f"  ({arguments.permutations} permutations per event)" if arguments.permutations > 0 else
        f"rows: train {train_features.shape}, val {val_features.shape}  (NO augmentation -- control)")

  train_matrix = xgb.DMatrix(train_features, label=train_labels)
  val_matrix = xgb.DMatrix(val_features, label=val_labels)
  history = {}
  model = xgb.train(
    {
      'objective': 'multi:softprob', 'num_class': n_classes, 'eval_metric': 'mlogloss', 'eta': 0.08,
      'tree_method': 'hist', 'grow_policy': 'lossguide', 'max_leaves': 31, 'max_depth': 0,
      'min_child_weight': 40, 'seed': arguments.seed,
      'nthread': int(os.environ.get('SLURM_CPUS_PER_TASK', 4)),
    },
    train_matrix, num_boost_round=arguments.max_learners,
    evals=[(train_matrix, 'train'), (val_matrix, 'val')], evals_result=history, verbose_eval=False
  )

  # THE STOPPING RULE: the learner count where |val - train| is MINIMAL, in units of ln K.
  train_curve = np.array(history['train']['mlogloss']) / scale
  val_curve = np.array(history['val']['mlogloss']) / scale
  stopping = int(np.argmin(np.abs(val_curve - train_curve)))

  stage = (0, stopping + 1)
  picked = lambda p, y: -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)) / scale
  train_errors = picked(model.predict(train_matrix, iteration_range=stage), train_labels)
  val_errors = picked(model.predict(val_matrix, iteration_range=stage), val_labels)

  train_mean, val_mean = float(train_errors.mean()), float(val_errors.mean())
  diff = abs(train_mean - val_mean)
  # `err`, as the trainer defines it: the standard error of the reported (train + val) / 2.
  err = 0.5 * float(np.sqrt(train_errors.var() / train_errors.size + val_errors.var() / val_errors.size))
  level = 0.5 * (train_mean + val_mean)

  print(f"\n  stopped at {stopping + 1} learners (|val - train| minimal over {arguments.max_learners})")
  print(f"  train {train_mean:.5f}   val {val_mean:.5f}   level {level:.5f}")
  print(f"  diff {diff:.5f}   err {err:.5f}   diff+err {diff + err:.5f}")
  print(f"  compare: the DEEP SET at this design gave diff 0.0055, err 0.0025, diff+err 0.0079,")
  print(f"           and the worst design in the campaign gave diff 0.0083 -- which forced")
  print(f"           loss_precision = 8.0e-3 and criterion (d)'s bar to 0.080.")

  with open(arguments.output, "w") as f:
    json.dump({
      "design": chosen, "permutations": arguments.permutations, "n_events": arguments.n_events,
      "stopping": stopping + 1, "train": train_mean, "val": val_mean, "level": level,
      "diff": diff, "err": err, "slack": diff + err,
      "train_curve": train_curve.tolist(), "val_curve": val_curve.tolist(),
    }, f, indent=2, default=float)
  print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
  main()
