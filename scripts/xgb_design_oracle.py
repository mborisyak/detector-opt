#!/usr/bin/env python3
"""An XGBoost ORACLE over the DESIGN SPACE, fitted to the 1M corpus, used to re-score BO trajectories.

WHAT THIS IS NOT. `detopt/bo/gbdt.py` fits a GBDT to a detector's EVENTS at one design, as a cheap
stand-in for the per-design regressor. This fits a GBDT to (design -> converged loss) pairs ACROSS
designs, so it is a surrogate for the whole BO objective, not for the regressor inside it.

WHY RE-SCORE AT ALL. Every campaign reports the loss its own network reached, so a trajectory from a
w2x-width run and one from a baseline-width run are on different vertical scales and their designs
cannot be compared directly. Scoring every banked design through ONE oracle puts them on a single
yardstick and asks only "how good is the design this arm found", with the network capacity and the
growth ladder divided out.

THE YARDSTICK BELONGS TO THE TRAINING CORPUS. The oracle is fitted to `ship-addr-prec1e2` -- baseline
width, OLD growth ladder (n0 2048). Re-scored numbers are therefore that campaign's loss level, not
the level of the run being re-scored. Only the ORDERING of designs transfers.

FEATURES ARE THE NOMINAL DESIGN, NOT THE SCALED CUBE. `intersection_penalty` and the sequential
stereo encoding map the same scaled coordinates to different physical stations, and `angle_only` has
a 1-d cube entirely, so the scaled space is not shared. The five physical numbers are.

TWO SUPPORT PROBLEMS ARE HANDLED EXPLICITLY, because the corpus does not cover what the newer
detectors propose:

* REORDERED STATIONS ARE CANONICALISED, NOT EXTRAPOLATED. Permuting two stations' z within one side
  of the magnet is the SAME detector -- the C solver sorts layers by z, and every station carries an
  identical `[0, +a, -a, 0]` structure with the same intra-station offsets -- so sorting within side
  maps a reordered design onto its physical twin, which the corpus does cover.
* OVERLAP AND NEGATIVE ANGLES ARE EXTRAPOLATION AND ARE FLAGGED. The corpus's minimum station gap is
  exactly the encoding's 100 cm floor, so it holds ZERO overlapping designs, and its angles never
  leave [0, 0.2]. A tree ensemble is piecewise-constant outside its support and would silently return
  the nearest in-support value. Such designs are marked and counted per tree; they are NOT dropped,
  because dropping them would flatter the very arms that proposed them.

PROTOCOL. 5 outer folds; for each, 3 folds train, 1 val, 1 test. The val fold does double duty --
it selects the hyper-parameters and it stops the boosting at its own minimum (`best_iteration`) --
and the test fold is touched once, after both choices are made. The final re-scoring model is refit
on all designs at the median selected round count.

    python scripts/xgb_design_oracle.py --out output/gbdt-oracle
"""

import argparse
import copy
import glob
import itertools
import json
import os

import numpy as np
import xgboost as xgb

CORPUS = "output/ship-cern/ship-addr-prec1e2"
N_STATIONS_UPSTREAM = 2
STATION_WIDTH = 100.0

GRID = {
  "max_depth": [2, 3, 4, 6],
  "learning_rate": [0.03, 0.06, 0.1],
  "min_child_weight": [1, 4],
  "subsample": [0.8, 1.0],
  "colsample_bytree": [0.8, 1.0],
}


def canonicalise(designs):
  """Sort station z within each side of the magnet, leaving the angle alone.

  A permutation of stations inside one side is the same physical detector, so this maps a reordered
  design onto the twin the corpus covers rather than treating it as a new point."""
  d = np.array(designs, dtype=np.float64, copy=True)
  d[:, :N_STATIONS_UPSTREAM] = np.sort(d[:, :N_STATIONS_UPSTREAM], axis=1)
  d[:, N_STATIONS_UPSTREAM:-1] = np.sort(d[:, N_STATIONS_UPSTREAM:-1], axis=1)
  return d


def total_overlap(designs):
  """Total pairwise intersection of the station z footprints, in cm, per design."""
  z = np.asarray(designs)[:, :-1]
  separation = np.abs(z[:, :, None] - z[:, None, :])
  pairwise = np.maximum(STATION_WIDTH - separation, 0.0)
  rows, columns = np.triu_indices(z.shape[1], k=1)
  return pairwise[:, rows, columns].sum(axis=1)


def read_cells(tree):
  """``[(seed, arm, payload)]`` for every cell of a tree that banked at least one design."""
  out = []
  for path in sorted(glob.glob(os.path.join(tree, "*", "*", "results.json"))):
    seed, arm = path.split(os.sep)[-3], path.split(os.sep)[-2]
    if "." in arm:
      continue
    payload = json.load(open(path))
    rows = payload["results"] if isinstance(payload, dict) else payload
    if len(rows) > 0:
      out.append((seed, arm, payload, rows))
  return out


def nominal_designs(payload, rows):
  """Rows -> ``(n, 5)`` nominal designs, reconstructing the fixed stations of a 1-dof angle design."""
  designs = [r["design"] for r in rows]
  width = max(len(d) for d in designs)
  if width == 5:
    return np.asarray(designs, dtype=np.float64)
  detector = payload.get("config", {}).get("detector", {}) if isinstance(payload, dict) else {}
  fixed = None
  for arguments in detector.values():
    if isinstance(arguments, dict) and arguments.get("fixed_stations") is not None:
      fixed = np.asarray(arguments["fixed_stations"], dtype=np.float64)
  if fixed is None:
    raise ValueError("a 1-dof design needs `fixed_stations` in the stored config to be re-scored")
  return np.array([list(fixed) + [float(d[-1])] for d in designs], dtype=np.float64)


def fit_fold(train, validation, parameters, n_rounds, early):
  """Fit one booster, stopped at the VALIDATION minimum; returns ``(booster, best_iteration)``."""
  model = xgb.XGBRegressor(
    n_estimators=n_rounds, early_stopping_rounds=early, eval_metric="rmse",
    tree_method="hist", verbosity=0, **parameters
  )
  model.fit(train[0], train[1], eval_set=[(validation[0], validation[1])], verbose=False)
  return model, int(model.best_iteration)


def cross_validate(X, y, seed, n_rounds, early):
  """5 folds; per fold 3 train / 1 val (tuning + early stop) / 1 test (touched once)."""
  rng = np.random.default_rng(seed)
  fold = rng.permutation(len(y)) % 5
  grid = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
  results = []
  for k in range(5):
    test_mask = fold == k
    validation_mask = fold == (k + 1) % 5
    train_mask = ~(test_mask | validation_mask)
    train = (X[train_mask], y[train_mask])
    validation = (X[validation_mask], y[validation_mask])
    best = None
    for parameters in grid:
      model, rounds = fit_fold(train, validation, parameters, n_rounds, early)
      score = float(np.sqrt(np.mean((model.predict(validation[0]) - validation[1]) ** 2)))
      if best is None or score < best[0]:
        best = (score, parameters, rounds, model)
    validation_rmse, parameters, rounds, model = best
    predicted = model.predict(X[test_mask])
    truth = y[test_mask]
    rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
    baseline = float(np.sqrt(np.mean((truth - train[1].mean()) ** 2)))
    order = float(np.corrcoef(predicted, truth)[0, 1])
    results.append({
      "fold": k, "n_train": int(train_mask.sum()), "n_validation": int(validation_mask.sum()),
      "n_test": int(test_mask.sum()), "parameters": parameters, "best_iteration": rounds,
      "validation_rmse": validation_rmse, "test_rmse": rmse, "mean_predictor_rmse": baseline,
      "test_pearson": order,
    })
  return results


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--corpus", default=CORPUS)
  parser.add_argument("--trees", nargs="*", default=None, help="trees to re-score; default every tree beside the corpus")
  parser.add_argument("--out", default="output/gbdt-oracle")
  parser.add_argument("--rescored", default=None, help="where the re-scored trees are written")
  parser.add_argument("--seed", type=int, default=123456)
  parser.add_argument("--n-rounds", type=int, default=2000)
  parser.add_argument("--early-stopping", type=int, default=50)
  arguments = parser.parse_args()
  os.makedirs(arguments.out, exist_ok=True)
  rescored_root = arguments.rescored if arguments.rescored is not None else os.path.join(arguments.out, "rescored")

  designs, losses = [], []
  for _seed, _arm, payload, rows in read_cells(arguments.corpus):
    for row, design in zip(rows, nominal_designs(payload, rows)):
      if row.get("loss") is None:
        continue
      designs.append(design)
      losses.append(float(row["loss"]))
  X = canonicalise(np.asarray(designs))
  y = np.asarray(losses)
  print(f"[corpus] {arguments.corpus}: {len(y)} designs, loss {y.min():.4f}-{y.max():.4f}, sd {y.std():.4f}")
  support = {
    "angle": (float(X[:, -1].min()), float(X[:, -1].max())),
    "station": [(float(X[:, i].min()), float(X[:, i].max())) for i in range(X.shape[1] - 1)],
    "max_overlap_cm": float(total_overlap(X).max()),
  }
  print(f"[support] angle {support['angle']}, max overlap {support['max_overlap_cm']:.1f} cm")

  folds = cross_validate(X, y, arguments.seed, arguments.n_rounds, arguments.early_stopping)
  test = np.array([f["test_rmse"] for f in folds])
  base = np.array([f["mean_predictor_rmse"] for f in folds])
  print("\n[cv] 3 train / 1 val / 1 test, val does tuning AND early stopping")
  for f in folds:
    print(f"  fold {f['fold']}  n={f['n_train']}/{f['n_validation']}/{f['n_test']}  "
          f"rounds={f['best_iteration']:4d}  val_rmse={f['validation_rmse']:.4f}  "
          f"test_rmse={f['test_rmse']:.4f}  mean_pred={f['mean_predictor_rmse']:.4f}  r={f['test_pearson']:.3f}")
  print(f"  TEST RMSE {test.mean():.4f} +/- {test.std(ddof=1)/np.sqrt(len(test)):.4f}   "
        f"mean-predictor {base.mean():.4f}   skill {100*(1-test.mean()/base.mean()):.1f}%")

  parameters = min(folds, key=lambda f: f["validation_rmse"])["parameters"]
  rounds = int(np.median([f["best_iteration"] for f in folds]))
  final = xgb.XGBRegressor(n_estimators=max(rounds, 1), tree_method="hist", verbosity=0, **parameters)
  final.fit(X, y, verbose=False)
  print(f"\n[final] refit on all {len(y)} designs, {rounds} rounds, parameters {parameters}")

  trees = arguments.trees
  if trees is None:
    root = os.path.dirname(arguments.corpus.rstrip(os.sep))
    trees = sorted(d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d))
  summary = {"corpus": arguments.corpus, "support": support, "folds": folds,
             "final_parameters": parameters, "final_rounds": rounds, "trees": {}}
  print("\n[rescore] design counts and how much of each tree the oracle actually covers")
  for tree in trees:
    cells = read_cells(tree)
    if len(cells) == 0:
      continue
    n_total = n_out = 0
    for seed, arm, payload, rows in cells:
      nominal = nominal_designs(payload, rows)
      canonical = canonicalise(nominal)
      overlap = total_overlap(canonical)
      angle = canonical[:, -1]
      outside = (overlap > 0) | (angle < support["angle"][0]) | (angle > support["angle"][1])
      predicted = final.predict(canonical)
      new_rows = []
      for row, value, flag, over in zip(rows, predicted, outside, overlap):
        row = copy.deepcopy(row)
        row["measured_loss"] = row.get("loss")
        row["loss"] = float(value)
        row["oracle_extrapolated"] = bool(flag)
        row["station_overlap_cm"] = float(over)
        new_rows.append(row)
      out_payload = copy.deepcopy(payload) if isinstance(payload, dict) else {"results": None}
      out_payload["results"] = new_rows
      out_payload["completed"] = True
      out_payload["best_loss"] = float(min(r["loss"] for r in new_rows))
      destination = os.path.join(rescored_root, os.path.basename(tree.rstrip(os.sep)), seed, arm)
      os.makedirs(destination, exist_ok=True)
      json.dump(out_payload, open(os.path.join(destination, "results.json"), "w"))
      n_total += len(new_rows)
      n_out += int(outside.sum())
    summary["trees"][os.path.basename(tree.rstrip(os.sep))] = {"designs": n_total, "extrapolated": n_out}
    print(f"  {os.path.basename(tree.rstrip(os.sep)):28s} {n_total:4d} designs  "
          f"{n_out:4d} extrapolated ({100*n_out/max(n_total,1):5.1f}%)")

  json.dump(summary, open(os.path.join(arguments.out, "oracle.json"), "w"), indent=2)
  print(f"\n[out] {os.path.join(arguments.out, 'oracle.json')}")
  print(f"[out] re-scored trees under {rescored_root}")
  print(f"[next] python scripts/plot_median.py {rescored_root}/<tree> --out output/plots --name gbdt_rescored.png")


if __name__ == "__main__":
  main()
