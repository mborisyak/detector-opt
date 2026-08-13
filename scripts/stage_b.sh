#!/usr/bin/env bash
# STAGE B: sweep the read-out noise at the operating point Stage A chose.
#
#     ./scripts/stage_b.sh extremes m4 r32           # the winning cell, named
#     ./scripts/stage_b.sh mmsym m6 r16 --dry-run
#
# GOVERNING DOCUMENT: docs/tuning-preregistration.md section 3, Stage B. The arms were fixed there
# before any of this ran:
#
#     extremes   0.0125, 0.025, 0.05 (canonical), 0.10   -- BIDIRECTIONAL
#     mm-sym     0.05 (canonical), 0.10, 0.15            -- UPWARD ONLY
#
# `mm-sym` is upward-only because its lower end is set by the integrator, not by plausibility: below
# ~0.025 the tolerance the 25% rule implies sits under what the box can deliver and the guard fires on
# designs inside the box. `extremes` has no such floor at its own box and sweeps both ways.
#
# WHY STAGE B MIGHT NOT BE NEEDED AT ALL, and that is a real possibility rather than a hedge: Stage A
# measured the READINGS lever doing what noise was being held in reserve to do -- more readings widen
# the baseline-to-best span (0.2455 -> 0.3659 at m=4) and take criterion (d)'s bar from 32.6% to 21.9%
# of it. If the winning cell already clears the conditions, this sweep buys margin rather than
# feasibility, and a day of wall clock is a real price for margin. Decide before launching.
#
# TOLERANCE FOLLOWS NOISE IN BOTH DIRECTIONS. `integration_tolerance` is a fixed fraction of the
# read-out noise, so every arm carries its own -- TIGHTENING as well as loosening. Applying the
# coupling only where it helps is what `gate-audit.md` 3.3 failed an earlier config for.
# `extremes` keeps the stricter 10% coupling (its guard has never fired); `mm-sym` uses 25%.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/home/max/opt/pyenv/versions/3.11.9/envs/py3/bin/python}
[ -x "$PY" ] || { echo "stage_b: interpreter $PY not found" >&2; exit 1; }
CPUS=${CPUS:-2}

CANDIDATE=${1:?candidate: extremes | mmsym}
M_TOKEN=${2:?batch size as mN, e.g. m4}
R_TOKEN=${3:?readings as rN, e.g. r32}
DRY_RUN=0
[ "${4:-}" = "--dry-run" ] && DRY_RUN=1

M=${M_TOKEN#m}
READS=${R_TOKEN#r}
OUT=output/screen

case "$CANDIDATE" in
  extremes)
    CONFIG=config/detector/enzyme_extremes.yaml
    ROOT=enzyme_inhib
    PROVENANCE=$OUT/provenance-extremes.md
    CEILING="--ceiling 1.0"
    PRECISION=${PRECISION:-8.0e-3}
    TOTAL_STEPS=1280
    COUPLING=0.10
    NOISES="0.0125 0.025 0.05 0.10"
    ;;
  mmsym)
    CONFIG=config/detector/enzyme_mm_sym.yaml
    ROOT=enzyme_mm
    PROVENANCE=$OUT/provenance-mm-sym.md
    CEILING=""
    PRECISION=${PRECISION:-1.0e-2}
    TOTAL_STEPS=2560
    COUPLING=0.25
    NOISES="0.05 0.10 0.15"
    ;;
  *) echo "stage_b: unknown candidate $CANDIDATE" >&2; exit 1 ;;
esac

[ -f "$CONFIG" ] || { echo "stage_b: $CONFIG does not exist" >&2; exit 1; }
[ -f "$PROVENANCE" ] || { echo "stage_b: $PROVENANCE does not exist -- the gate is declared, not inferred" >&2; exit 1; }

# Steps are held constant in TOTAL across the sweep, exactly as in Stage A: dt = duration /
# (n_measurements * n_steps_per_measurement), so a sweep that fixed steps PER MEASUREMENT would vary
# integration accuracy and cost along with the lever under test.
STEPS=$(( TOTAL_STEPS / READS ))

echo "stage B: ${CANDIDATE} at m=${M}, ${READS} readings, ${STEPS} steps/measurement (${TOTAL_STEPS} total)"
submitted=0
for noise in $NOISES; do
  # awk, NOT python: every Python invocation on this box goes through SLURM, including trivial ones,
  # and a launcher that needs the scheduler to compute a product is a launcher that cannot run.
  tolerance=$(awk -v c="$COUPLING" -v n="$noise" 'BEGIN { printf "%.6g", c * n }')
  label="${CANDIDATE}-${M_TOKEN}-${R_TOKEN}-n${noise}"
  if [ -f "$OUT/${label}.json" ]; then
    echo "  skip  ${label} (already screened)"
    continue
  fi
  command="env OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} \
NUMEXPR_NUM_THREADS=${CPUS} XLA_FLAGS=--xla_force_host_platform_device_count=1 \
${PY} scripts/screen_task.py --config ${CONFIG} --label ${label} --provenance ${PROVENANCE} \
--m ${M} --set ${ROOT}.n_measurements=${READS} --set ${ROOT}.n_steps_per_measurement=${STEPS} \
--set ${ROOT}.measurement_noise=${noise} --set ${ROOT}.integration_tolerance=${tolerance} \
--n-designs 512 --n-events 16384 --seeds 10 --iters 20 40 ${CEILING} \
--loss-precision ${PRECISION} --output ${OUT}/${label}.json"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  would submit ${label} (noise ${noise}, tolerance ${tolerance})"
  else
    # --time from Stage A's MEASURED costs (6.3-15.2 s/design x ~1312 scorings); a user cannot extend
    # a running job's limit, so under-estimating it loses the whole cell.
    id=$(sbatch --parsable --cpus-per-task="${CPUS}" --mem-per-cpu=1500 --time=20:00:00 \
         -J "scb-${label}" -o "${OUT}/${label}.log" --wrap="${command}")
    echo "  job ${id}  ${label}  (noise ${noise}, tolerance ${tolerance})"
  fi
  submitted=$((submitted + 1))
done
echo
echo "stage B: ${submitted} screens"
echo "judge:   ${PY} scripts/stage_a_verdict.py --candidate ${CANDIDATE}   (same rule, section 6)"
echo
echo "⚠️  --loss-precision ${PRECISION} is a PLACEHOLDER unless it was measured AT THIS CELL. Criterion"
echo "    (d) is re-judged from the stored curves once the real value exists -- no re-run needed."
