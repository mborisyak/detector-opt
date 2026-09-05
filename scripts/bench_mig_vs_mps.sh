#!/bin/bash
# MIG against MPS on one box: does hard isolation beat sharing for these cells?
#
#   scripts/bench_mig_vs_mps.sh [seconds]     # default 300
#
# THE QUESTION. The campaign runs 6 cells per GPU under MPS, where they time-slice the same SMs. Do
# they suppress each other badly enough that 4 hard-isolated MIG instances -- 46 SMs each, guaranteed --
# get more done? Two competing stories both predict 100% "utilisation", so that metric cannot separate
# them; per-cell THROUGHPUT can.
#
#   4 MIG / GPU   ->  8 concurrent, isolated       each cell owns 46 SMs
#   4 MPS / GPU   ->  8 concurrent, shared         equal-N control: isolation is the ONLY difference
#   6 MPS / GPU   -> 12 concurrent, shared         what the campaign runs today
#
# The equal-N control is the point. Comparing MIG-8 against MPS-12 alone would confound isolation with
# concurrency, and either could win for the wrong reason.
#
# WHY A FIXED WINDOW AND NOT COMPLETED DESIGNS. One design costs ~1.5e5 detector calls and takes ~16
# minutes under contention, so waiting for completions would make this an hour-long test per arm and
# leave it hostage to where each cell happened to be in its growth ladder. Instead every cell is given
# a budget it CANNOT exhaust, run for a fixed wall time, and scored on the pool it consumed. Same
# workload in every arm, so the ratio is what carries meaning -- not the absolute number.
#
# ⚠️ EACH ARM NEEDS A GPU MODE SWITCH, which requires no CUDA process attached. The script stops
# everything between arms and verifies the switch took; a silently-failed switch would otherwise report
# one configuration under another's name.
set -u
SECONDS_PER_ARM="${1:-300}"
REPO=/root/detector-opt
BENCH=/tmp/bench
PY=/root/venv/bin/python
CFG=intersect
STRATEGY=intersect-from_scratch-norewind
BUDGET=2097152                # the campaign's own budget -- see below
# ⛔️ DO NOT RAISE THIS TO MAKE CELLS "UNREACHABLE". `training.budget` SIZES THE PREALLOCATED EVENT
# POOLS, it is not only a stopping rule. A first attempt used 134217728 to guarantee no cell finished;
# every cell then tried to allocate a 16 GiB buffer, which does not fit a 23.62 GiB MIG instance
# alongside the model, and all 8 spun retrying the allocation for the whole window while reporting
# nothing. Cells do not finish inside the window anyway -- one design costs ~16 minutes under
# contention -- so the campaign budget already gives the behaviour the unreachable one was meant to.

cd "$REPO" || exit 90

cleanup() {
  pkill -f "scripts/bo.py" 2>/dev/null
  sleep 4
  pkill -9 -f "scripts/bo.py" 2>/dev/null
  sleep 2
}

mig_off() {
  systemctl stop nvidia-mps 2>/dev/null
  for i in 0 1; do nvidia-smi mig -i $i -dci >/dev/null 2>&1; nvidia-smi mig -i $i -dgi >/dev/null 2>&1; done
  for i in 0 1; do nvidia-smi -i $i -mig 0 >/dev/null 2>&1; done
  sleep 3
  nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | tr '\n' ' '
}

mig_on() {
  systemctl stop nvidia-mps 2>/dev/null
  for i in 0 1; do nvidia-smi -i $i -mig 1 >/dev/null 2>&1; done
  sleep 3
  for i in 0 1; do nvidia-smi mig -i $i -cgi 14,14,14,14 -C >/dev/null 2>&1; done
  sleep 2
  nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | tr '\n' ' '
}

launch() {   # $1 = tag, $2 = index, $3 = CUDA_VISIBLE_DEVICES value
  local out="$BENCH/$1/cell$2"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$3" XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
    setsid nohup "$PY" scripts/bo.py "=$CFG" "strategy=$STRATEGY" "output=$out" \
      "seed=$((1000 + $2))" "training.budget=$BUDGET" \
      > "$out/cell.log" 2>&1 < /dev/null &
}

score() {    # $1 = tag, $2 = n cells, $3 = elapsed seconds
  "$PY" - "$BENCH/$1" "$2" "$3" <<'PY'
import sys, glob, os, re
root, n, secs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
used = []
for d in sorted(glob.glob(os.path.join(root, "cell*"))):
    log = os.path.join(d, "cell.log")
    if not os.path.exists(log): continue
    last = None
    for line in open(log, errors="ignore"):
        m = re.search(r"pool\s+(\d+)/(\d+)", line)
        if m: last = int(m.group(1))
    if last: used.append(last)
used.sort()
if not used:
    print(f"  {os.path.basename(root):<10} NO CELL REPORTED A POOL -- check {root}/cell0/cell.log")
    sys.exit(0)
per = [u / secs for u in used]
tot = sum(per)
print(f"  {os.path.basename(root):<10} cells={len(used)}/{n}  per-cell {sum(per)/len(per):8.1f} ev/s "
      f"(min {per[0]:7.1f}, max {per[-1]:7.1f})   AGGREGATE {tot:9.1f} ev/s")
PY
}

run_arm() {  # $1 = tag, $2 = n, $3.. = device list
  local tag=$1; shift; local n=$1; shift
  local devs=("$@")
  rm -rf "$BENCH/$tag"; mkdir -p "$BENCH/$tag"
  echo "  [$tag] launching $n cells..."
  for ((i = 0; i < n; i++)); do launch "$tag" "$i" "${devs[$((i % ${#devs[@]}))]}"; done
  sleep "$SECONDS_PER_ARM"
  local alive; alive=$(pgrep -fc "scripts/bo.py" 2>/dev/null || echo 0)
  echo "  [$tag] alive at cutoff: $alive/$n"
  score "$tag" "$n" "$SECONDS_PER_ARM"
  cleanup
}

echo "=== $SECONDS_PER_ARM s per arm, workload: $STRATEGY on $CFG ==="
cleanup

echo "--- arm 1: 4 MIG per GPU (8 concurrent, isolated) ---"
echo "  mig mode: $(mig_on)"
mapfile -t MIGS < <(nvidia-smi -L | grep -oE "MIG-[0-9a-f-]+")
echo "  MIG devices found: ${#MIGS[@]}"
[ "${#MIGS[@]}" -eq 8 ] && run_arm mig8 8 "${MIGS[@]}" || echo "  SKIPPED: expected 8 MIG devices"

echo "--- arm 2: 4 MPS per GPU (8 concurrent, shared) ---"
echo "  mig mode: $(mig_off)"
systemctl start nvidia-mps 2>/dev/null; sleep 4
echo "  mps: $(systemctl is-active nvidia-mps)"
run_arm mps8 8 0 1

echo "--- arm 3: 6 MPS per GPU (12 concurrent, shared) ---"
echo "  mps: $(systemctl is-active nvidia-mps)"
run_arm mps12 12 0 1

echo "=== done. Higher AGGREGATE wins on throughput; higher per-cell wins on latency. ==="
