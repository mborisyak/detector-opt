#!/usr/bin/env python3
"""What does `meta` actually report about a design, compared with `from_scratch`?

    python scripts/capacity_probe.py --n-warmup 5 --n-repeats 5 --output output/capacity/probe.json

The benchmark compares training strategies by the loss each reports for the designs it visits, but
the two arms never visit the same designs, so their numbers have never been compared on equal terms.
This does that directly (design due to the user):

    1. draw `n_warmup` random designs plus one TARGET design;
    2. run the continual trainer over the warm-up designs, so its network carries their history;
    3. train it on the TARGET  -> what `meta` would report for that design;
    4. train a fresh network on the SAME target -> what `from_scratch` would report;
    5. repeat, so the variation is measured rather than assumed.

The hypotheses it separates, which the run data cannot:

  * **trivial exit.** `meta` stops after ~5.5k calls against `from_scratch`'s ~28k, and the stopping
    rule tests the train-val GAP, not the loss level -- a network that is not fitting the new window
    has train ~ val for free. If that is all that is happening, meta's level on the target will be
    WORSE at its much smaller spend.
  * **capacity.** The continual network must fit every design it has seen at once. Fitting designs
    individually is easier than fitting them conditionally, so an under-powered network degrades as
    history accumulates -- visible as meta's level worsening with `n_warmup`.
  * **transfer.** History genuinely helps, and meta reaches the same level for less data. Then its
    level matches or beats from_scratch's.

Reported per repeat: the loss LEVEL, the reported uncertainty (|val - train| + hypot(sems)), and the
detector calls spent -- and across repeats their spread, which is what decides whether any of the
single-seed differences seen so far mean anything.
"""
import argparse
import copy
import json
import os

import numpy as np

import detopt.detector
import detopt.utils.io
import detopt.utils.config
from detopt.nn.trainer import ContinualTrainer, DesignTrainer

from capacity_sweep import VARIANTS


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="enzyme")
  parser.add_argument("--n-warmup", type=int, default=5, help="designs meta sees before the target")
  parser.add_argument("--n-repeats", type=int, default=5)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--target-from", default=None,
                      help="results.json to draw the TARGET designs from, best-first, instead of "
                           "drawing them uniformly. Uniform targets land in the middle of the range "
                           "(measured: 32-65%% of the way from the 1/3 ceiling to the best design), "
                           "but the offset that matters is the one where BO ranks -- among the best "
                           "designs, whose gaps are ~0.005. Warm-up designs stay random: the point "
                           "is that meta's network carries a history, not which history.")
  parser.add_argument("--variant", default=None,
                      help="substring of a scripts/capacity_sweep.py variant label; its regressor "
                           "REPLACES the config's. Use to ask whether a higher-capacity network "
                           "removes meta's reported-level bias.")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  # `config/<name>.yaml` names its detector by REFERENCE (`detector: enzyme`), which the gearup
  # entry point resolves and a direct load does not -- so the detector config is loaded from its own
  # file, exactly as scripts/bo_gbdt.py does.
  config = detopt.utils.config.load_config(f"config/{arguments.config}.yaml")
  detector = detopt.detector.from_config(
      detopt.utils.config.load_config(f"config/detector/{config['detector']}.yaml"))
  dimension = int(detector.design_dim())

  if arguments.variant is not None:
    matched = [v for v in VARIANTS if arguments.variant in v[0]]
    if len(matched) != 1:
      raise SystemExit(f"--variant {arguments.variant!r} matched {len(matched)} labels: "
                       f"{[v[0] for v in matched]}")
    label, regressor = matched[0]
    config["regressor"] = copy.deepcopy(regressor)
    print(f"regressor: {label.strip()}  {regressor}", flush=True)

  targets = None
  if arguments.target_from is not None:
    with open(arguments.target_from) as f:
      ranked = sorted(detopt.utils.io.complete_results(json.load(f)["results"]), key=lambda r: r["loss"])
    targets = [(np.asarray(r["x_scaled"], np.float32), float(r["loss"])) for r in ranked]
    print("targets (best-first): " + ", ".join(f"{loss:.4f}" for _, loss in targets[:arguments.n_repeats]),
          flush=True)

  rows = []
  for repeat in range(arguments.n_repeats):
    rng = np.random.default_rng(1000 + repeat)
    designs = rng.random((arguments.n_warmup + 1, dimension)).astype(np.float32)
    target = designs[-1]
    if targets is not None:
      target = targets[repeat % len(targets)][0]
      designs[-1] = target
    seeds = np.random.SeedSequence(arguments.seed + repeat)

    # (2)+(3) meta: the continual network sees the warm-up designs, then the target.
    meta = ContinualTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed + repeat)
    history = []
    for step, design in enumerate(designs[:-1]):
      result = meta.train(design, int(seeds.spawn(1)[0].generate_state(1)[0]), step=step)
      if result is None:
        break
      history.append(float(result.objective_loss))
    meta_result = meta.train(target, int(seeds.spawn(1)[0].generate_state(1)[0]), step=arguments.n_warmup)

    # (4) from_scratch: a fresh network on the SAME target, nothing carried.
    scratch = DesignTrainer.from_config(detector, config, checkpoint_dir=None, seed=arguments.seed + repeat)
    scratch_result = scratch.train(target, int(seeds.spawn(1)[0].generate_state(1)[0]), step=0)

    row = {
        "repeat": repeat,
        "warmup_losses": history,
        "meta": None if meta_result is None else {
            "loss": float(meta_result.objective_loss), "std": float(meta_result.objective_std),
            "spent": int(meta_result.spent)},
        "from_scratch": None if scratch_result is None else {
            "loss": float(scratch_result.objective_loss), "std": float(scratch_result.objective_std),
            "spent": int(scratch_result.spent)},
    }
    rows.append(row)
    m, s = row["meta"], row["from_scratch"]
    if m is not None and s is not None:
      print(f"repeat {repeat}: meta {m['loss']:.4f} +/-{m['std']:.4f} ({m['spent']:6d} calls) | "
            f"from_scratch {s['loss']:.4f} +/-{s['std']:.4f} ({s['spent']:6d} calls) | "
            f"meta - scratch = {m['loss'] - s['loss']:+.4f}", flush=True)

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump({"n_warmup": arguments.n_warmup, "rows": rows}, f, indent=2)

  pairs = [(r["meta"], r["from_scratch"]) for r in rows if r["meta"] and r["from_scratch"]]
  if len(pairs) > 0:
    difference = np.array([m["loss"] - s["loss"] for m, s in pairs])
    print(f"\nON THE SAME DESIGNS, {len(pairs)} repeats:")
    print(f"  meta         level {np.median([m['loss'] for m, _ in pairs]):.4f}  "
          f"spread {np.std([m['loss'] for m, _ in pairs]):.4f}  "
          f"spend {np.median([m['spent'] for m, _ in pairs]):.0f}")
    print(f"  from_scratch level {np.median([s['loss'] for _, s in pairs]):.4f}  "
          f"spread {np.std([s['loss'] for _, s in pairs]):.4f}  "
          f"spend {np.median([s['spent'] for _, s in pairs]):.0f}")
    print(f"  meta - from_scratch: median {np.median(difference):+.4f}, "
          f"meta worse on {int((difference > 0).sum())}/{len(difference)}")


if __name__ == "__main__":
  main()
