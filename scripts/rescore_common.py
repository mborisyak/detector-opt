#!/usr/bin/env python3
"""Re-score every arm's designs with ONE instrument, so their curves can be compared.

    python scripts/rescore_common.py --runs output/neural-confirm/meta output/neural-confirm/from_scratch \\
        --max-epochs 60 --budget 32768 --output output/rescore/neural-confirm.json

Why this exists. `scripts/capacity_probe.py` measured that `meta` reports **+0.0103 more than
`from_scratch` on identical designs** (4/4 repeats), i.e. the two arms do not report on the same
scale. The benchmark's headline compares their curves, and that difference (`meta` 0.0755 against
`from_scratch` 0.0701 on the verified numbers) is SMALLER than the offset, in the direction that
makes `meta` look worse.

`scripts/verify_trajectory.py` does not remove it: it restores the network the run reported a design
with (its per-design checkpoint) and continues training THAT network on fresh data, so it removes the
optimiser's self-selection, not the trainer's scale. It does fine-tune on a single design, which may
wash some of the offset out -- how much is an empirical question this script answers rather than
assumes.

The instrument here is deliberately the dumbest thing that is identical for every design of every
arm: `FullBudgetTrainer` -- a FRESHLY initialised regressor, the whole budget sampled up front at the
fixed design, exactly `max_epochs` epochs of cosine-decayed training, no checkpoint restored, no
warm start, no data growing, no early exit. Whatever it reports, it reports the same way everywhere,
so a difference between two arms under it is a difference between their DESIGNS.

Read the OFFSET line. If `meta`'s reported-minus-rescored is systematically above `from_scratch`'s,
the run numbers carry an arm-level scale difference and the benchmark's arm comparison is measuring
the trainer. If the rescored curves keep the reported ordering, the ordering is about the designs.

Cost is `n_points` x `arms` full-budget trainings; keep `budget`/`max_epochs` modest -- what matters
is that they are the SAME for every point, not that they are large.
"""
import argparse
import json
import os

import numpy as np

import detopt.detector
import detopt.utils.config
from detopt.nn.trainer import FullBudgetTrainer
from detopt.utils import io


def incumbents(results, n_max):
  """The iterations where best-so-far improved, thinned to at most `n_max` uniform in calls, the
  last always kept -- the same selection `verify_trajectory.py` makes, so the two are comparable."""
  reported = np.asarray([r["loss"] for r in results], np.float64)
  calls = np.cumsum([int(r["spent"]) for r in results]).astype(np.float64)
  improves = np.flatnonzero(reported < np.minimum.accumulate(np.concatenate([[np.inf], reported[:-1]])))
  if improves.shape[0] <= n_max:
    return [int(i) for i in improves]
  targets = np.linspace(calls[improves][0], calls[improves][-1], n_max)
  chosen = {int(improves[np.argmin(np.abs(calls[improves] - t))]) for t in targets}
  chosen.add(int(improves[-1]))
  return sorted(chosen)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--runs", nargs="+", required=True, help="run directories (or results.json paths)")
  parser.add_argument("--config", default="enzyme")
  parser.add_argument("--n-points", type=int, default=6, help="incumbents re-scored per arm")
  parser.add_argument("--budget", type=int, default=32768, help="detector calls per re-score (same for all)")
  parser.add_argument("--max-epochs", type=int, default=60)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  config = detopt.utils.config.load_config(f"config/{arguments.config}.yaml")
  detector = detopt.detector.from_config(
      detopt.utils.config.load_config(f"config/detector/{config['detector']}.yaml"))
  # One instrument: the same budget and the same number of epochs for every design of every arm.
  config["training"] = dict(config["training"], budget=arguments.budget)

  rows = []
  for run in arguments.runs:
    path = run if run.endswith(".json") else os.path.join(run, "results.json")
    with open(path) as f:
      results = io.check_bo_results(json.load(f)["results"], path)
    selected = incumbents(results, arguments.n_points)
    print(f"\n=== {path}: {len(results)} designs, re-scoring incumbents {selected}", flush=True)
    for index in selected:
      record = results[index]
      # A fresh trainer per design: the pools are filled to capacity per call, so reusing one
      # instance would let an earlier design's events into a later design's window.
      trainer = FullBudgetTrainer.from_config(detector, config, max_epochs=arguments.max_epochs,
                                              seed=arguments.seed)
      result = trainer.train(np.asarray(record["x_scaled"], np.float32), arguments.seed)
      row = {"run": path, "iteration": int(record["iteration"]),
             "reported": float(record["loss"]), "reported_std": float(record["loss_std"]),
             "rescored": float(result.objective_loss), "rescored_std": float(result.objective_std),
             "reported_spent": int(record["spent"])}
      rows.append(row)
      os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
      with open(arguments.output, "w") as f:
        json.dump({"budget": arguments.budget, "max_epochs": arguments.max_epochs, "rows": rows}, f, indent=2)
      print(f"  iter {row['iteration']:3d}: reported {row['reported']:.4f} -> "
            f"rescored {row['rescored']:.4f} ({row['rescored'] - row['reported']:+.4f})", flush=True)

  print(f"\n{'run':44s} {'reported best':>14s} {'rescored best':>14s} {'median offset':>14s}")
  for run in arguments.runs:
    path = run if run.endswith(".json") else os.path.join(run, "results.json")
    got = [r for r in rows if r["run"] == path]
    if len(got) == 0:
      continue
    offset = np.median([r["rescored"] - r["reported"] for r in got])
    print(f"{path:44s} {min(r['reported'] for r in got):14.4f} "
          f"{min(r['rescored'] for r in got):14.4f} {offset:+14.4f}")
  print("\nThe offset column is reported-scale minus common-scale. Arms differing here were not "
        "being compared on the same instrument; the 'rescored best' column is.")


if __name__ == "__main__":
  main()
