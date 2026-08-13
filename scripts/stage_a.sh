#!/usr/bin/env bash
# STAGE A of the benchmark tuning round: `m` x `n_measurements` at fixed noise, both candidates.
#
#     ./scripts/stage_a.sh              # submit everything
#     ./scripts/stage_a.sh --dry-run    # print the commands and submit nothing
#
# GOVERNING DOCUMENT: docs/tuning-preregistration.md section 3. The grid, the bars, the iteration
# pairs and the decision rule were all fixed there BEFORE any of this ran, and nothing here may be
# changed in response to a result. If a setting needs to move, amend that document first and say why.
#
# WHY A SCRIPT RATHER THAN A SHELL LOOP: every screen is its own SLURM job, so a screen that dies at
# hour three does not take the other seventeen with it, and the scheduler -- not this script -- decides
# how many run at once. At 2 CPUs each on an 11-CPU box that is five concurrent with the rest queued,
# which is the intended behaviour.
#
# THE INTERPRETER IS EXPLICIT AND THAT MATTERS. Under `srun`, plain `python` resolves to a pyenv shim
# with no xgboost and `screen_task.py` dies at import. Threads are pinned for the same reason the
# scripts pin them internally: SLURM says which cores a job may use, not how many threads it should
# start, and several jobs each spawning a dozen BLAS threads is how this box reached load 44.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/max/opt/pyenv/versions/3.11.9/envs/py3/bin/python
CPUS=${CPUS:-2}
DRY_RUN=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    # Release one candidate without the other. Used deliberately: `mm-sym`'s step count was still open
    # when `extremes` was ready, and screening a task at one setting while its campaign runs at another
    # makes the two incomparable.
    --only) ONLY="${2:?--only needs a candidate name}"; shift 2 ;;
    *) echo "stage_a: unknown argument $1" >&2; exit 1 ;;
  esac
done

N_DESIGNS=${N_DESIGNS:-512}
N_EVENTS=${N_EVENTS:-16384}
SEEDS=${SEEDS:-10}
# ONE run to 40; 5->10, 10->20 and 20->40 are all read off the stored curves afterwards. See the
# pre-registration section 4: the headline is 10->20, 20->40 is the high-n reading, and 5->10 is the
# declared relaxation that may never be reported as a plain PASS.
ITERS="20 40"

OUT=output/screen
mkdir -p "$OUT"

# INTEGRATION STEPS ARE HELD CONSTANT IN TOTAL, NOT PER MEASUREMENT.
#
# `dt = duration / (n_measurements * n_steps_per_measurement)`, so the readings lever ALREADY refines
# the integration. Holding `n_steps_per_measurement` fixed across the sweep would make the 32-reading
# arm 4x more accurate AND 4x more expensive than the 8-reading arm -- confounding the lever under
# test with integration accuracy, and paying for the privilege. Holding the TOTAL fixed gives every
# arm the same `dt`, the same truncation error, and the same cost, so the only thing that varies is
# the thing being swept.
#
# The totals come from the campaign-scale guard scan (2e6 events, worst box corner), which is where
# the guard binds -- it is a max over sampled events and grows with the count, so a margin measured at
# screening scale is not a margin:
#
#   mm-sym  (tolerance 1.25e-2)   1280 steps 0.85x FIRES | 1920 2.31x | 2560 4.21x | 3840 9.50x
#   extremes(tolerance 5.0e-3)     640 steps 1.62x       | 1280 8.83x | 1920 20.4x | 2560 36.6x
#
# Chosen: mm-sym 2560 (4.21x) and extremes 1280 (8.83x). The shipped values -- 1280 and 640 -- are
# NOT taken: mm-sym's fires outright past ~2.6e5 events, and extremes' 1.62x is too thin to survive a
# statistic that grows with sample size. This became possible only after the RKC2 increment-form fix;
# before it, refining past 1280 total steps made the error WORSE.
declare -A TOTAL_STEPS=( [extremes]=1280 [mmsym]=2560 )

# candidate | detector config | config root key | provenance | ceiling flag | loss-precision
#
# `--ceiling 1.0` for `extremes` only: cross-entropy divided by ln(n_classes) has an EXACT
# no-information level, so measuring it would only add noise. `mm-sym` is a regression target and
# measures its own.
#
# LOSS PRECISION. `extremes` 8.0e-3 is INHERITED from addendum-inhibitor-precision.md and has never
# been measured on the BINARY task. `mm-sym` has no measurement at all; 1.0e-2 below is a DECLARED
# PLACEHOLDER, not a measurement. Neither is load-bearing for what Stage A decides: `loss_precision`
# enters only criterion (d), and because the screener now stores every seed's full best-so-far curve,
# (d) can be re-judged at the measured value later by arithmetic, with no re-run. No verdict may be
# claimed for either candidate until the real value is measured per pre-registration section 5 --
# across >= 8 designs x >= 2 seeds, taking the MAXIMUM.
CANDIDATES=(
  "extremes|config/detector/enzyme_extremes.yaml|enzyme_inhib|$OUT/provenance-extremes.md|--ceiling 1.0|8.0e-3"
  "mmsym|config/detector/enzyme_mm_sym.yaml|enzyme_mm|$OUT/provenance-mm-sym.md||1.0e-2"
)

# Batch sizes. `mm-sym` starts at 3 because the task has p = 3 target parameters and m < p is
# underdetermined BY CONSTRUCTION -- a property of the task, not a tuning choice.
declare -A BATCH_SIZES=( [extremes]="4 6 8" [mmsym]="3 4 6" )
READINGS="8 16 32"

submitted=0
for row in "${CANDIDATES[@]}"; do
  IFS='|' read -r name config root provenance ceiling precision <<< "$row"
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then
    continue
  fi
  [ -f "$config" ] || { echo "stage_a: $config does not exist" >&2; exit 1; }
  [ -f "$provenance" ] || { echo "stage_a: $provenance does not exist -- the gate is declared, not inferred" >&2; exit 1; }

  for m in ${BATCH_SIZES[$name]}; do
    for reads in $READINGS; do
      label="${name}-m${m}-r${reads}"
      if [ -f "$OUT/${label}.json" ]; then
        echo "  skip  ${label} (already screened)"
        continue
      fi
      steps=$(( TOTAL_STEPS[$name] / reads ))
      command="env OMP_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} \
NUMEXPR_NUM_THREADS=${CPUS} XLA_FLAGS=--xla_force_host_platform_device_count=1 \
${PY} scripts/screen_task.py --config ${config} --label ${label} --provenance ${provenance} \
--m ${m} --set ${root}.n_measurements=${reads} --set ${root}.n_steps_per_measurement=${steps} \
--n-designs ${N_DESIGNS} --n-events ${N_EVENTS} \
--seeds ${SEEDS} --iters ${ITERS} ${ceiling} --loss-precision ${precision} \
--output ${OUT}/${label}.json"
      if [ "$DRY_RUN" -eq 1 ]; then
        echo "  would submit ${label}:"
        echo "    ${command}"
      else
        # MEMORY IS REQUESTED EXPLICITLY, and that is not a detail. SLURM's default here is
        # 2000 MB/CPU, so a 2-CPU screen reserves 4 GB while MEASURING 1.5-1.6 GB of RSS. Five of them
        # exhaust the node's 20 GB declaration, AllocMem hits RealMemory, and with no time limits to
        # backfill against nothing else can schedule at all -- including the GPU jobs on the critical
        # path -- while the machine itself sits with 19 GB available. It is a bookkeeping deadlock,
        # not memory pressure. 1500 MB/CPU = 3 GB/job leaves headroom over the measured peak and
        # leaves the node schedulable.
        # --time IS REQUIRED (backfill will not schedule an unlimited job out of order) and must be
        # SIZED FROM MEASURED COST, not guessed. A cell is ~1312 design scorings: 512 for the
        # landscape plus 800 for the iteration test (10 seeds x 40 iterations, BO and random arms).
        # MEASURED s/design: 6.3 at m=4/r8, 11.2-11.5 at m=4/r16-32, 13.1-15.2 at m=6. So m=6 runs
        # 5-6 h and m=8 approaches 9-10 h. An earlier 8 h limit was set from a stale 2.19 s/design
        # figure and would have killed five cells an hour short of finishing -- and a user CANNOT
        # extend a running job's limit, only an admin can, so the only remedy was to lose the work.
        # 20 h is generous against the worst measured cell; the cost of over-generosity here is only
        # weaker backfill, and the cost of under-generosity is the whole cell.
        id=$(sbatch --parsable --cpus-per-task="${CPUS}" --mem-per-cpu=1500 --time=20:00:00 \
             -J "scr-${label}" -o "${OUT}/${label}.log" --wrap="${command}")
        echo "  job ${id}  ${label}"
      fi
      submitted=$((submitted + 1))
    done
  done
done

echo
echo "stage A: ${submitted} screens (${DRY_RUN} = dry run)"
echo "watch:   squeue -o '%.6i %.22j %.8T %.10M'"
echo "judge:   ${PY} scripts/compare_screens.py ${OUT}/extremes-m*.json"
echo
echo "NOTE: every screen runs its iteration test ONCE to 40 and stores each seed's full best-so-far"
echo "curve, so 5->10, 10->20 and 20->40 all come off the same run. Results are NOT comparable with"
echo "screens on disk from before 2026-08-13: the RKC2 arrangement changed, and the event offset is"
echo "derived from the total iteration count so a run to 40 truncated at 20 is not a run to 20."
