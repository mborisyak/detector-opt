#!/usr/bin/env python3
"""Build a VERIFIED twin of a campaign tree for plot_median.py: for every cell ``<seed>/<arm>`` of the source tree
that carries ``verified.txt``, write a results.json whose per-design ``loss`` is the verification's held-out
``test_loss`` (matched by point index to the rows with a loss, and checked against the point's ``reported_loss``);
everything else in the payload is copied, so the budget, the spend and the completion flag are the run's own.
Cells without ``verified.txt`` are skipped, so a partial source gives a partial twin and plot_median's ``--paired``
reads the arms on one seed set.

    python scripts/verified_tree.py output/linear/d1n2/test /tmp/linear-verified/d1n2
"""
import json
import os
import sys


def build(src, dst):
  written, skipped = 0, 0
  for seed in sorted(os.listdir(src)):
    for arm in sorted(os.listdir(os.path.join(src, seed))):
      cell = os.path.join(src, seed, arm)
      if not os.path.exists(os.path.join(cell, "verified.txt")) or not os.path.exists(os.path.join(cell, "results.json")):
        skipped += 1
        continue
      payload = json.load(open(os.path.join(cell, "results.json")))
      points = {int(p["point"]): p for p in json.load(open(os.path.join(cell, "verification.json")))["points"]}
      rows = [r for r in payload["results"] if r.get("loss") is not None]
      if len(points) != len(rows):
        raise SystemExit(f"{cell}: {len(points)} verification points for {len(rows)} designs")
      for i, row in enumerate(rows):
        p = points[i]
        if abs(float(p["reported_loss"]) - float(row["loss"])) > 1e-9:
          raise SystemExit(f"{cell}: point {i} reported_loss {p['reported_loss']} != row loss {row['loss']}")
        row["loss"] = float(p["test_loss"])
      payload["best_loss"] = min(r["loss"] for r in rows)
      out = os.path.join(dst, seed, arm)
      os.makedirs(out, exist_ok=True)
      json.dump(payload, open(os.path.join(out, "results.json"), "w"))
      for marker in ("done.txt", "verified.txt"):
        open(os.path.join(out, marker), "w").close()
      written += 1
  return written, skipped


if __name__ == "__main__":
  w, s = build(sys.argv[1], sys.argv[2])
  print(f"{sys.argv[2]}: {w} verified cells written, {s} skipped")
