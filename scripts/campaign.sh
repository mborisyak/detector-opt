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
# THE ACCELERATOR REQUEST, AND ITS DEFAULT WAS MISSING -- this line is the fix. The header above has
# always documented "GRES (env, default `shard:1`)", but nothing ever assigned it, so an unset GRES
# made `${GRES:+--gres=${GRES}}` expand to NOTHING and the job was submitted with no shard at all.
# It still runs: jax finds no visible device, falls back to the host, and a campaign that was meant
# to be on the GPU grinds through on CPU while `squeue -o %b` reports N/A. Nothing raises, so the
# only symptom is a campaign that is inexplicably slow and whose numbers are not the ones intended.
#
# `${GRES-shard:1}` USES `-`, NOT `:-`, AND THE DIFFERENCE IS THE WHOLE CONTRACT. `:-` substitutes
# for unset OR EMPTY, which would override the documented escape hatch for a `device: cpu` task
# (`GRES= ./scripts/campaign.sh ...`, header line 34) and queue it behind a shard it never uses. `-`
# substitutes only when GRES is UNSET, so an explicitly empty GRES stays empty.
GRES=${GRES-shard:1}
# Sized just ABOVE the measured run length -- see the --time comment at the sbatch call. Measured:
# 0.8-1.2 h per run at budget 2097152 with per-epoch plotting off.
TIME_LIMIT=${TIME_LIMIT:-01:45:00}
# gearup `key=value` overrides passed through to bo.py, e.g. OVERRIDES='training.budget=3300000'.
# Without this every campaign variant needs its own config file, which is how the rehearsal had to
# work around it.
OVERRIDES=${OVERRIDES:-}
# THE JOB-NAME PREFIX, AND IT IS A CORRECTNESS SETTING, NOT COSMETICS. The duplicate guard below
# recognises an in-flight run by the SLURM job name alone, which carries the arm and the seed but
# not the task -- so two campaigns on DIFFERENT tasks sharing a seed and an arm look identical to
# it, and the second one silently skips runs it never submitted. Give each campaign its own prefix
# (`NAME_PREFIX=mm ./scripts/campaign.sh ...`) whenever another campaign may be on the queue.
NAME_PREFIX=${NAME_PREFIX:-camp}

# DRIVER (env, default `scripts/bo.py`) is the BO driver to run.
DRIVER=${DRIVER:-scripts/bo.py}

# Seeds are DERIVED, not listed: the same `random.Random(SUPER_SEED)` draw the Snakefile uses, so the
# two drivers agree seed-for-seed and a comparison against earlier campaigns is paired rather than
# accidental. Extending a campaign means asking for more seeds, which appends the next draws -- never
# typing extra values in, because an invented seed looks identical to a paired one at the call site.
SUPER_SEED=${SUPER_SEED:-123456}
mapfile -t ALL_SEEDS < <(python3 -c 'import random, sys
rng = random.Random(int(sys.argv[1]))
print("\n".join(str(rng.randint(0, 2 ** 31 - 1)) for _ in range(int(sys.argv[2]))))' "$SUPER_SEED" "$SEEDS")

echo "campaign: config=${CONFIG}  arms=${ARMS[*]}  seeds=${SEEDS}  -> ${OUT}"
for ((i = 0; i < SEEDS; i++)); do
  seed=${ALL_SEEDS[$i]}
  for arm in "${ARMS[@]}"; do
    run="${OUT}/${seed}/${arm}"
    # COMPLETION IS THE `completed` FIELD, NOT THE FILE'S EXISTENCE, and the difference silently cost
    # a campaign cell before this was fixed. The comment here used to read "`results.json` EXISTS ONLY
    # FOR A FINISHED RUN -- bo.py writes `partial.json` while in flight and renames at the end", and
    # that has not been true since bo.py moved to one trajectory file: it now REWRITES `results.json`
    # after every single design, carrying `completed: false` until the budget pool empties. So a run
    # killed after one design leaves a `results.json` behind, the old test read it as finished, and
    # the resubmission skipped a cell that had 1 design in it instead of the 32 it was meant to have.
    # Nothing failed and nothing warned -- the campaign was simply one arm short.
    #
    # `bo.py` states this field is there "for readers that only have the file", which is exactly this
    # reader. A missing or unreadable file is NOT complete, so a run with no results at all resubmits.
    if [ "$("${PY}" -c "import json,sys
try:
  print(json.load(open(sys.argv[1])).get('completed') is True)
except Exception:
  print(False)" "${run}/results.json" 2>/dev/null)" = "True" ]; then
      echo "  skip  ${seed}/${arm} (already complete)"
      continue
    fi
    # DUPLICATE GUARD. Without it, re-invoking while a run is queued submits a SECOND job with the
    # same output directory: two `bo.py` processes writing one results.json and one orbax tree.
    # MEASURED in rehearsal -- a re-invocation duplicated all four in-flight runs.
    if squeue -h -o "%j" -u "$USER" | grep -qx "${NAME_PREFIX}-${arm}-${seed}"; then
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
      -J "${NAME_PREFIX}-${arm}-${seed}" -o "${run}/run.log" \
      --wrap="env XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=${CPUS} \
OPENBLAS_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} NUMEXPR_NUM_THREADS=${CPUS} \
${PY} -u ${DRIVER} =${CONFIG} output=${run} seed=${seed} nn_init_strategy=${arm} ${OVERRIDES}")
    echo "  job ${id}  ${seed}/${arm}"
  done
done
echo
echo "watch:   squeue -o '%.6i %.18j %.8T %.10M'"
echo "results: ${OUT}/<seed>/<arm>/results.json"
