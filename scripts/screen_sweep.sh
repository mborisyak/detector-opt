#!/bin/bash
# Screen VARIANTS OF THE PROBLEM against the brief -- a smooth benchmark that rewards iterations --
# without running an optimiser on any of them.
#
#   ./scripts/screen_sweep.sh OUT [n_designs] [n_events] [n_anchors] [jobs]
#
# Each variant is one `enzyme_landscape.py screen`: a random sample of the landscape (how much of it
# is plateau, what a shotgun of k draws gets) plus a semivariogram (how far a design moves before its
# loss decorrelates). ~300 evaluations, minutes, against the ~1500 a BO campaign costs -- so this
# decides WHICH settings deserve a campaign, and the campaign then confirms or refutes it.
#
# One knob moves per variant, from the config default, because two knobs at once cannot be told
# apart afterwards. Every variant is a DIFFERENT FUNCTION: the ones that move the melting prior also
# move the normaliser, so only the C columns of the table compare across rows -- that is what they
# are there for.
set -euo pipefail

OUT=${1:-output/screen}
N_DESIGNS=${2:-192}
N_EVENTS=${3:-8192}
N_ANCHORS=${4:-12}
JOBS=${5:-6}

cd "$(dirname "$0")/.."
export PYTHONPATH=${PYTHONPATH:-$PWD}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_FLAGS=--xla_force_host_platform_device_count=1

mkdir -p "$OUT"

# variant | overrides. `base` is the config as it stands, and every other row is one step from it.
VARIANTS=(
  "base       |"
  # The batch size: the design is 2m-dimensional, so this is the benchmark's dimension knob AND its
  # information knob at once -- more experiments means both a harder search and a better achievable
  # answer. A realistic assay plate runs anything from a pair of conditions to a dozen.
  "m2         |--set enzyme.n_experiments=2"
  "m6         |--set enzyme.n_experiments=6"
  "m8         |--set enzyme.n_experiments=8"
  # The admissible set. Wider leaves more room to place experiments badly (so more for an optimiser
  # to win) at the cost of reintroducing the uninformative plateau; the screen measures both.
  "box_wide   |--set enzyme.temperature_bounds=[25.0,80.0]"
  "box_full   |--set enzyme.temperature_bounds=[0.0,100.0]"
  "box_tight  |--set enzyme.temperature_bounds=[42.0,62.0]"
  # The read-out. 5% of the concentration scale is the default; a plate reader doing an absorbance
  # assay is good to ~1-2% and bad to ~10%, so both directions stay realistic.
  "noise_lo   |--set enzyme.measurement_noise=0.02"
  "noise_hi   |--set enzyme.measurement_noise=0.10"
  # Fewer read-outs per experiment: each experiment then says less on its own, so how the BATCH is
  # arranged matters more. Eight points over an hour is a comfortable plate schedule; two is a
  # start/end pair, which is also what a manual assay gives.
  "reads2     |--set enzyme.n_measurements=2"
  # The enzyme population. Hexokinase melting points across isoforms and species spread wider than
  # the default 13 C window; [40, 65] is still inside the liquid range and still inside the box, and
  # it makes a larger share of the design space informative. This CHANGES THE OBJECTIVE (the target's
  # prior is also its normaliser), so it is comparable to the others only in C.
  "prior_wide |--set enzyme.parameters.T_melting=[40.0,65.0]"
  "prior_tight|--set enzyme.parameters.T_melting=[48.0,55.0]"
  # How long the assay runs. The turnover is calibrated so half conversion falls in 0.25-2 h, so at
  # a quarter of an hour most enzymes are caught mid-curve and the enzyme STOCK FRACTION starts to
  # matter (it sets how far the reaction gets); at four hours nearly everything has run to
  # completion and the read-out flattens. Both are ordinary assay lengths.
  "dur_short  |--set enzyme.duration=0.25"
  "dur_long   |--set enzyme.duration=4.0"
)

run_one() {
  IFS='|' read -r name overrides <<< "$1"
  name=${name//[[:space:]]/}
  local out="$OUT/$name.json"
  if [ -f "$out" ]; then echo "skip   $name (exists)"; return; fi
  # shellcheck disable=SC2086  # the overrides are deliberately word-split into argv
  python scripts/enzyme_landscape.py screen --n-designs "$N_DESIGNS" --n-events "$N_EVENTS" \
    --n-anchors "$N_ANCHORS" $overrides --output "$out" > "$OUT/$name.log" 2>&1
  echo "$name -> $(grep VERDICT "$OUT/$name.log" || echo FAILED)"
}
export -f run_one
export OUT N_DESIGNS N_EVENTS N_ANCHORS

printf '%s\n' "${VARIANTS[@]}" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

python scripts/plot_gbdt_study.py screens --directory "$OUT" | tee "$OUT/table.txt"
echo "screen sweep done -> $OUT"
