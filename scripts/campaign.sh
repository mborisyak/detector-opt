#!/usr/bin/env bash
# Launch the NEURAL campaign for one task: every (seed, strategy) pair as its own SLURM job.
#
#   ./scripts/campaign.sh CONFIG OUTPUT_ROOT [SEEDS] [ARMS...]
#   ./scripts/campaign.sh enzyme_inhib output/campaign-inhib 5 meta from_scratch
#
# CONFIG is a run config under config/ (the gearup root token), e.g. `enzyme_inhib` for
# config/enzyme_inhib.yaml. Each job gets ONE GPU shard, so with two shards SLURM runs two at a time
# and queues the rest -- which is exactly the behaviour wanted: no oversubscription, no manual
# babysitting, and a killed job can be resubmitted alone.
#
# WHY ONE JOB PER (seed, arm) rather than one job looping over them: a campaign that dies at hour six
# should not take the five finished runs with it. bo.py already skips a completed results.json, so a
# resubmission of the whole grid costs only what is missing.
#
# THE 8-HOUR FIGURE (docs/benchmark-acceptance.md 2.3) is about this command: 5 seeds x 2 arms is 10
# runs over 2 shards = 5 per shard, so a run must average ~1.6 h. Check that against the measured
# per-design cost BEFORE launching:
#
#     wall_clock_per_run ~ designs_per_run * seconds_per_design
#     designs_per_run    ~ budget / calls_per_design        (both from a short calibration run)
#
# If it does not fit, shrink the TASK's budget or drop an arm -- do not cut seeds below 5, and do not
# shorten the screening that established the settings.
#
# CPUS (env, default 4) is the per-job CPU request. Training runs on the GPU and needs few host
# threads, so 2 is enough to let a campaign share the node with CPU-bound screening jobs; SLURM queues
# whatever does not fit rather than oversubscribing, which is the whole point of going through it.
#
# GRES (env, default `shard:1`) is the accelerator request. Set it EMPTY for a task whose run config
# declares `device: cpu`, so it runs beside the GPU work instead of queueing behind a shard it will
# never use:
#
#     GRES= CPUS=2 TIME_LIMIT=00:30:00 ./scripts/campaign.sh <task> output/campaign-<task> 5
set -euo pipefail

CONFIG=${1:?config name, e.g. enzyme_inhib}
OUT=${2:?output root, e.g. output/campaign-inhib}
SEEDS=${3:-5}
shift 3 2>/dev/null || shift $#
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(meta from_scratch)

cd "$(dirname "$0")/.."
[ -f "config/${CONFIG}.yaml" ] || { echo "campaign: config/${CONFIG}.yaml does not exist" >&2; exit 1; }
mkdir -p "$OUT"

# THE INTERPRETER IS PINNED, and the reason is not pedantry. A bare `python` in the wrap line resolved
# correctly in testing ONLY because `PYENV_VERSION=py3` happened to be exported in the launching
# shell -- it is not in ~/.bashrc. Under a clean `bash -lc` there is no `python` on PATH at all, and
# with .bashrc sourced but no PYENV_VERSION it resolves to pyenv's global 3.13.9, which carries
# jax 0.8.1 instead of py3's 0.10.2. A campaign that silently runs on a different jax is worse than
# one that fails to start.
PY=${PY:-/home/max/opt/pyenv/versions/3.11.9/envs/py3/bin/python}
[ -x "$PY" ] || { echo "campaign: interpreter $PY not found or not executable" >&2; exit 1; }

# CPUS: training runs on the GPU and needs few host threads. 2 leaves room for CPU-bound screening
# jobs to share the node; SLURM queues what does not fit rather than oversubscribing.
CPUS=${CPUS:-2}
MEM_PER_CPU=${MEM_PER_CPU:-1500}
# Sized just ABOVE the measured run length -- see the --time comment at the sbatch call. Measured:
# 0.8-1.2 h per run at budget 2097152 with per-epoch plotting off.
TIME_LIMIT=${TIME_LIMIT:-01:45:00}
# gearup `key=value` overrides passed through to bo.py, e.g. OVERRIDES='training.budget=3300000'.
# Without this every campaign variant needs its own config file, which is how the rehearsal had to
# work around it.
OVERRIDES=${OVERRIDES:-}

# Seeds are drawn from a fixed list rather than 0..N-1: these are the seeds the earlier enzyme
# campaigns used, so a comparison against those runs is paired rather than accidental.
ALL_SEEDS=(1244111331 126382657 750143450 9577242 330924253 42 43 44 45 46)

echo "campaign: config=${CONFIG}  arms=${ARMS[*]}  seeds=${SEEDS}  -> ${OUT}"
for ((i = 0; i < SEEDS; i++)); do
  seed=${ALL_SEEDS[$i]}
  for arm in "${ARMS[@]}"; do
    run="${OUT}/${seed}/${arm}"
    # `results.json` EXISTS ONLY FOR A FINISHED RUN -- bo.py writes `partial.json` while in flight and
    # renames at the end -- so its presence is the completion test, no field to read.
    if [ -f "${run}/results.json" ]; then
      echo "  skip  ${seed}/${arm} (already complete)"
      continue
    fi
    # DUPLICATE GUARD. Without it, re-invoking while a run is queued submits a SECOND job with the
    # same output directory: two `bo.py` processes writing one results.json and one orbax tree.
    # MEASURED in rehearsal -- a re-invocation duplicated all four in-flight runs.
    if squeue -h -o "%j" -u "$USER" | grep -qx "camp-${arm}-${seed}"; then
      echo "  skip  ${seed}/${arm} (already queued or running)"
      continue
    fi
    # A CRASHED RUN IS RESUMED, and its `partial.json` is therefore LEFT WHERE IT IS. `bo.py`
    # restarts an interrupted run at the design boundary from that file plus the state pair it
    # commits between designs, so the scored designs are read back rather than re-measured; moving it
    # aside (which this used to do to results.json, when there was no resume) would throw that away
    # and pay for every design again.
    #
    # A run killed by an UNCONVERGEABLE DESIGN still repeats: the seed is fixed, so the resumed run
    # proposes the same design and fails on it again. That is a task setting to fix, not something a
    # relaunch can clear -- the failing design is on the record as an `incomplete` row.
    if [ -f "${run}/partial.json" ]; then
      [ -f "${run}/run.log" ] && mv "${run}/run.log" "${run}/run.$(date +%Y%m%d-%H%M%S).log"
      echo "  note  ${seed}/${arm}: previous run INCOMPLETE -- resuming from its committed state"
    fi
    mkdir -p "$run"
    # --time IS REQUIRED, AND MUST BE TIGHT. With no limit SLURM's backfill scheduler cannot compute
    # when resources free and will not start a job out of order: MEASURED in rehearsal, campaign jobs
    # sat PENDING for 4h05m with a CPU free and BOTH GPU shards idle, while an identical job with
    # --time=00:05:00 started in 45 s. Over-generous limits fail the same way -- 02:00:00 refused to
    # backfill where 00:25:00 started in 38 s. Size it just above the measured run length.
    # --mem-per-cpu likewise: the 2000 MB/CPU default reserves 8 GB per job at CPUS=4, and two of
    # those exhaust the node's 20 GB declaration and deadlock every other job on the box.
    id=$(sbatch --parsable ${GRES:+--gres=${GRES}} --cpus-per-task="${CPUS}" --mem-per-cpu="${MEM_PER_CPU}" \
      --time="${TIME_LIMIT}" \
      -J "camp-${arm}-${seed}" -o "${run}/run.log" \
      --wrap="env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${CPUS} \
OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} NUMEXPR_NUM_THREADS=${CPUS} \
${PY} -u scripts/bo.py =${CONFIG} output=${run} seed=${seed} nn_init_strategy=${arm} ${OVERRIDES}")
    echo "  job ${id}  ${seed}/${arm}"
  done
done
echo
echo "watch:   squeue -o '%.6i %.18j %.8T %.10M'"
echo "results: ${OUT}/<seed>/<arm>/results.json"
