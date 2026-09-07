"""Partial results of the fresh SHiP campaign: per task, strategy and regime -- cells done, designs per run,
best reported loss (penalised) and, where verified, best held-out loss; plus the strategy comparison at norewind."""
import glob
import json
import os

import numpy as np

root = "/home/max/dev/detector-opt/output/ship-cern-fresh"
STRATEGIES = ("from_scratch", "continue", "meta")
REGIMES = ("norewind", "rewind-01", "rewind-025", "sp-l03-s1e2", "sp-l06-s1e2")
for task in ("intersect", "angle"):
  print(f"\n### {task}")
  print(f"{'strategy':<13}{'regime':<13}{'done':>5}{'designs':>9}{'calls/design':>14}{'best reported':>15}{'best verified':>15}")
  table = {}
  for strategy in STRATEGIES:
    for regime in REGIMES:
      cells = sorted(glob.glob(f"{root}/{task}/select/*/{strategy}/{regime}/results.json"))
      done = [c for c in cells if os.path.exists(os.path.join(os.path.dirname(c), "done.txt"))]
      designs, calls, bests, vbests = [], [], [], []
      for c in done:
        r = json.load(open(c))
        rows = [x for x in r["results"] if x.get("loss") is not None]
        designs.append(len(rows))
        calls.extend(x["spent"] for x in rows)
        bests.append(min(x["loss"] for x in rows) if len(rows) > 0 else np.nan)
        v = os.path.join(os.path.dirname(c), "verification.json")
        if os.path.exists(os.path.join(os.path.dirname(c), "verified.txt")):
          pts = json.load(open(v)).get("points", [])
          vbests.append(min(p["test_loss"] for p in pts) if len(pts) > 0 else np.nan)
      table[(strategy, regime)] = (len(done), designs, bests)
      if len(done) == 0:
        print(f"{strategy:<13}{regime:<13}{0:>5}")
        continue
      print(f"{strategy:<13}{regime:<13}{len(done):>5}{np.mean(designs):>9.1f}{np.mean(calls):>14.0f}{np.nanmean(bests):>15.4f}"
            f"{(np.nanmean(vbests) if len(vbests) > 0 else float('nan')):>15.4f}  n_verified={len(vbests)}")
  print("  strategies at each regime (mean best reported loss over done seeds):")
  for regime in REGIMES:
    parts = [f"{s}={np.nanmean(table[(s, regime)][2]):.4f}(n={table[(s, regime)][0]})" for s in STRATEGIES if table[(s, regime)][0] > 0]
    if len(parts) > 0:
      print(f"    {regime:<13} " + "  ".join(parts))
