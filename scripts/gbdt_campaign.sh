#!/bin/bash
# Campaign runner for the enzyme benchmark: a set of ARMS, each run over an ITERATION axis with no
# detector-call budget (see scripts/bo_gbdt.py), against a random-search NULL.
#
#   ./scripts/gbdt_campaign.sh ARMS_FILE OUT [bo_seeds] [null_seeds] [n_iterations] [jobs]
#
# The questions it answers, read off the run JSONs by `plot_gbdt_study.py acceptance`:
#
#   1. do the GP-driven iterations SEPARATE from the null -- does best-so-far keep falling as
#      iterations accumulate, faster than drawing the same number of designs at random?
#   2. are post-incumbent proposals CLOSER to the incumbent than two random points are to each
#      other -- i.e. is the GP localising, or is it guessing?
#
# ARMS_FILE holds one arm per line, `name|mode|flags`, with `#` comments and blank lines ignored:
#
#   ard  |bo    |--kernel ard-rbf
#   pi   |bo    |--kernel permutation-invariant-rbf
#   null |random|
#
# `mode` is bo, random, or both. There are NO built-in arms and no default flags: every difference
# between arms is written in the file, so what was compared is legible from the file that ran it.
# An arm declared `bo` alone borrows the `null` arm's random runs, and the analysis refuses the loan
# unless the two arms' recorded PROBLEM settings are identical -- which is what makes one null valid
# for several kernels but never for two different boxes, batch sizes or enzyme populations.
#
# Anything not stated by a flag comes from config/detector/enzyme.yaml AS IT IS NOW. Each run JSON
# records the settings it resolved to, so results carry their own provenance rather than inheriting
# whatever the config says on the day someone reads them.
set -euo pipefail

ARMS=${1:?arms file}
OUT=${2:?output directory}
BO_SEEDS=${3:-8}
NULL_SEEDS=${4:-10}
ITERS=${5:-60}
JOBS=${6:-8}

cd "$(dirname "$0")/.."
# The package is pip-installed editable against the MAIN checkout, so a run from a worktree imports
# the wrong tree unless PYTHONPATH points at this one.
export PYTHONPATH=${PYTHONPATH:-$PWD}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
# One thread per run: the runs are the parallelism, and oversubscribed BLAS/OpenMP inside each of
# them would only fight for the same cores.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_force_host_platform_device_count=1

mkdir -p "$OUT"
rm -f "$OUT/failures.txt"
cp "$ARMS" "$OUT/arms.txt"

jobs_file=$(mktemp)
trap 'rm -f "$jobs_file"' EXIT
while IFS='|' read -r name mode flags; do
  case "${name## }" in ""|\#*) continue;; esac
  # Trim the padding that keeps the arms file readable in columns. The name and the mode are single
  # words; the flags keep their internal spaces and lose only the outer ones.
  name=${name//[[:space:]]/}; mode=${mode//[[:space:]]/}
  flags=${flags:-}; flags=${flags#"${flags%%[![:space:]]*}"}; flags=${flags%"${flags##*[![:space:]]}"}
  for m in $([ "$mode" = both ] && echo "bo random" || echo "$mode"); do
    seeds=$([ "$m" = bo ] && echo "$BO_SEEDS" || echo "$NULL_SEEDS")
    for seed in $(seq 0 $((seeds - 1))); do echo "$name|$m|$seed|${flags:-}" >> "$jobs_file"; done
  done
done < "$ARMS"

run_one() {
  IFS='|' read -r arm mode seed flags <<< "$1"
  local out="$OUT/$arm-$mode-$seed.json"
  if [ -f "$out" ]; then echo "skip   $arm $mode seed $seed (exists)"; return; fi
  # shellcheck disable=SC2086  # the arm's flags are deliberately word-split into argv
  if ! python scripts/bo_gbdt.py --mode "$mode" --seed "$seed" --n-iterations "$ITERS" $flags \
      --output "$out" > "$OUT/$arm-$mode-$seed.log" 2>&1; then
    # A failed run writes no JSON, so without this it simply vanishes from the analysis and the
    # campaign still reports success. Record it where the summary below can count it.
    echo "FAILED $arm $mode seed $seed -- see $OUT/$arm-$mode-$seed.log" | tee -a "$OUT/failures.txt"
    return 0
  fi
  echo "$arm $mode seed $seed -> $(python -c "import json; d=json.load(open('$out')); \
    print(f\"{d['best_loss']:.5f} ({d['best_rmse_c']:.2f} C)\")")"
}
export -f run_one
export OUT ITERS

xargs -a "$jobs_file" -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

if [ -s "$OUT/failures.txt" ]; then
  echo "REFUSING to analyse: $(wc -l < "$OUT/failures.txt") run(s) failed --"; cat "$OUT/failures.txt"
  exit 1
fi
python scripts/plot_gbdt_study.py acceptance --directory "$OUT" --output "$OUT/acceptance.json"
python scripts/plot_gbdt_study.py convergence --directory "$OUT" --output "$OUT/convergence.png"
# The held-out curve only exists if the arms were scored twice; an arm carrying --no-verify has no
# `verified_loss`, and that is a legitimate way to run a campaign, not a reason to report failure
# after every run has already succeeded.
python scripts/plot_gbdt_study.py convergence --directory "$OUT" --key verified_loss \
  --output "$OUT/convergence_verified.png" || echo "skipped the held-out plot (runs have no verified_loss)"
echo "campaign done -> $OUT"
