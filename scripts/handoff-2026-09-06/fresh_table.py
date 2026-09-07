"""Fresh CERN SHiP campaign: self-reported against verified, like for like.

Reported loss = trained loss + design penalty (what BO minimised). Verified loss = the verification's held-out
test loss of the re-trained network + the SAME design penalty, so both carry the penalty. Per task, strategy and
regime (3 seeds): designs per run, calls per design, mean best reported, mean best verified, and the mean per-design
offset verified - reported with its SEM over all verified designs. Only cells with verified.txt enter the verified
columns."""
import glob
import json
import os

import numpy as np

root = "/home/max/dev/detector-opt/output/ship-cern-fresh"
STRATEGIES = ("from_scratch", "continue", "meta")
REGIMES = ("norewind", "rewind-01", "rewind-025", "sp-l03-s1e2", "sp-l06-s1e2")
for task in ("intersect", "angle"):
  print(f"\n### {task}   (best = min over a run's designs; means over seeds; offset = verified - reported per design)")
  print(f"{'strategy':<13}{'regime':<13}{'runs':>5}{'ver':>4}{'designs':>8}{'calls/des':>10}{'best rep':>10}{'best ver':>10}{'offset':>9}{'±sem':>7}{'n_des':>6}")
  rank = {}
  for strategy in STRATEGIES:
    for regime in REGIMES:
      cells = sorted(glob.glob(f"{root}/{task}/select/*/{strategy}/{regime}/results.json"))
      done = [c for c in cells if os.path.exists(os.path.join(os.path.dirname(c), "done.txt"))]
      designs, calls, best_rep, best_ver, offsets = [], [], [], [], []
      n_ver = 0
      for c in done:
        r = json.load(open(c))
        rows = [x for x in r["results"] if x.get("loss") is not None]
        designs.append(len(rows))
        calls.extend(x["spent"] for x in rows)
        best_rep.append(min(x["loss"] for x in rows))
        d = os.path.dirname(c)
        if not os.path.exists(os.path.join(d, "verified.txt")):
          continue
        n_ver += 1
        pts = {p["point"]: p for p in json.load(open(os.path.join(d, "verification.json")))["points"]}
        ver = []
        for i, x in enumerate(rows):
          if i in pts:
            penalty = x["loss"] - x["trained_loss"]
            v = pts[i]["test_loss"] + penalty
            ver.append(v)
            offsets.append(v - x["loss"])
        if len(ver) > 0:
          best_ver.append(min(ver))
      if len(done) == 0:
        continue
      off = np.asarray(offsets)
      rank[(strategy, regime)] = (np.mean(best_rep), np.mean(best_ver) if len(best_ver) else np.nan)
      print(f"{strategy:<13}{regime:<13}{len(done):>5}{n_ver:>4}{np.mean(designs):>8.1f}{np.mean(calls):>10.0f}{np.mean(best_rep):>10.4f}"
            f"{(np.mean(best_ver) if len(best_ver) else float('nan')):>10.4f}{(off.mean() if len(off) else float('nan')):>+9.4f}"
            f"{(off.std(ddof=1) / np.sqrt(len(off)) if len(off) > 1 else float('nan')):>7.4f}{len(off):>6}")
  print("  strategy order per regime, best reported | best verified:")
  for regime in REGIMES:
    rep = "  ".join(f"{s}={rank[(s, regime)][0]:.4f}" for s in STRATEGIES if (s, regime) in rank)
    ver = "  ".join(f"{s}={rank[(s, regime)][1]:.4f}" for s in STRATEGIES if (s, regime) in rank)
    print(f"    {regime:<13} rep: {rep}\n    {'':<13} ver: {ver}")
