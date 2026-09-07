#!/bin/bash
# Standard two-panel plots (plot_median.py --mean) of the fresh CERN campaign, self-reported and verified.
# plot_median reads trees of <seed>/<arm>/results.json, so per task and regime a tree is assembled in the job tmp dir:
#   reported : symlinks to the mirrored results.json
#   verified : copies with each design's loss replaced by test_loss + design penalty (verified cells only)
set -u
root=/home/max/dev/detector-opt
mirror=$root/output/ship-cern-fresh
trees=/home/max/dev/detector-opt/output/plots/fresh-2026-09-05/trees
out=$root/output/plots/fresh-2026-09-05
rm -rf "$trees"; mkdir -p "$out"
python3 - "$mirror" "$trees" <<'EOF'
import glob, json, os, sys
mirror, trees = sys.argv[1], sys.argv[2]
for c in sorted(glob.glob(f"{mirror}/*/select/*/*/*/results.json")):
  d = os.path.dirname(c)
  task, _, seed, strategy, regime = d.split("/")[-5:]
  if not os.path.exists(f"{d}/done.txt"):
    continue
  rep = f"{trees}/{task}/reported/{regime}/{seed}/{strategy}"
  os.makedirs(rep, exist_ok=True)
  os.symlink(c, f"{rep}/results.json")
  if not os.path.exists(f"{d}/verified.txt"):
    continue
  r = json.load(open(c))
  pts = {p["point"]: p for p in json.load(open(f"{d}/verification.json"))["points"]}
  rows = []
  for i, x in enumerate(r["results"]):
    if x.get("loss") is None or i not in pts:
      continue
    y = dict(x)
    y["loss"] = pts[i]["test_loss"] + (x["loss"] - x["trained_loss"])
    rows.append(y)
  r["results"] = rows
  r["best_loss"] = min(y["loss"] for y in rows)
  ver = f"{trees}/{task}/verified/{regime}/{seed}/{strategy}"
  os.makedirs(ver, exist_ok=True)
  json.dump(r, open(f"{ver}/results.json", "w"))
print("trees built")
EOF
cd "$root" || exit 90
for task in intersect angle; do
  for kind in reported verified; do
    python scripts/plot_median.py "$trees/$task/$kind/norewind" "$trees/$task/$kind/rewind-01" "$trees/$task/$kind/rewind-025" \
      "$trees/$task/$kind/sp-l03-s1e2" "$trees/$task/$kind/sp-l06-s1e2" --mean --out "$out" --name "${task}-${kind}.png" 2>&1 | grep -vE "Warning|warn" | tail -2
  done
done
ls -la --time-style=+%H:%M "$out" | awk 'NR>1{print $6, $7}'
