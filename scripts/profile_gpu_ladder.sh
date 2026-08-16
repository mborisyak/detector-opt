#!/usr/bin/env bash
# Aggregate GPU throughput against PROCESS COUNT, with and without MPS.
#
# Each worker is `profile_gpu_throughput.py run`: it loads a pre-sampled fixture and trains for a fixed
# wall-clock budget, so every process does identical GPU work and no time goes into simulating events.
# Aggregate throughput is the sum of the workers' steps/s. Without MPS, N processes TIME-SLICE one GPU
# and the sum stays flat; with MPS they co-run and it should rise until SMs or memory saturate.
#
# ⚠️ Under Default compute mode MPS is OPT-IN: a client that does not set CUDA_MPS_PIPE_DIRECTORY makes
# its own context and time-slices, with no error. Each rung therefore records whether an
# `nvidia-cuda-mps-server` process exists, so an MPS=on row that reports `server=no` is invalid on its
# face rather than quietly wrong.
#
#   MPS=off scripts/profile_gpu_ladder.sh "1 2 4 6 8"
set -u

LADDER=${1:-"1 2 4 8"}
MPS=${MPS:-on}
MPS_PIPE=${MPS_PIPE:-$HOME/.mps}
CONFIG=${CONFIG:-=enzyme_extremes}
SECONDS_PER=${SECONDS_PER:-60}
WORK=${WORK:-/tmp/claude-1000/-home-max-dev-detector-opt/f322215c-c143-48c0-ad31-e515686eef05/scratchpad/gpuladder}
DATASET=${DATASET:-${WORK}/../probe05.npz}
RESULT=${RESULT:-output/profile-gpu-${MPS}.tsv}

mkdir -p "${WORK}" "$(dirname "${RESULT}")"
printf 'n_procs\tmps\tserver\tagg_steps_per_s\tmed_steps_per_s\tsamples_per_s\tpeak_mib\tutil_pct\tfailures\n' > "${RESULT}"

if [ "${MPS}" = "on" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE}"
else
  unset CUDA_MPS_PIPE_DIRECTORY
fi
echo "MPS=${MPS} pipe=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} dataset=${DATASET} seconds=${SECONDS_PER}"
nvidia-smi --query-gpu=compute_mode,memory.total --format=csv,noheader

for n in ${LADDER}; do
  echo "=================== N = ${n} (MPS=${MPS}) ==================="
  rm -rf "${WORK}/n${n}"; mkdir -p "${WORK}/n${n}"

  ( while true; do
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep 2
    done ) > "${WORK}/n${n}/gpu.csv" 2>/dev/null &
  sampler=$!

  # Wait on the WORKER pids only: a bare `wait` also waits for the sampler, which never exits.
  pids=()
  for i in $(seq 1 "${n}"); do
    python -u scripts/profile_gpu_throughput.py run "${CONFIG}" \
      dataset="${DATASET}" seconds="${SECONDS_PER}" seed=$((100 + i)) \
      output="${WORK}/n${n}/w${i}.json" > "${WORK}/n${n}/w${i}.log" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  kill "${sampler}" 2>/dev/null; wait "${sampler}" 2>/dev/null

  server=$(pgrep -f "[n]vidia-cuda-mps-server" >/dev/null && echo yes || echo no)
  read -r agg med sam ok < <(python3 - "${WORK}/n${n}" <<'PY'
import glob, json, statistics, sys
rows = []
for p in glob.glob(sys.argv[1] + "/w*.json"):
    try: rows.append(json.load(open(p)))
    except Exception: pass
if rows:
    s = [r["steps_per_s"] for r in rows]
    print(f"{sum(s):.1f} {statistics.median(s):.1f} {sum(r['samples_per_s'] for r in rows):.0f} {len(rows)}")
else:
    print("0 0 0 0")
PY
)
  mem=$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END{print m}' "${WORK}/n${n}/gpu.csv")
  util=$(awk -F, '{gsub(/ /,"",$2); s+=$2; c++} END{if(c) printf "%.0f", s/c}' "${WORK}/n${n}/gpu.csv")
  fails=$((n - ok))

  echo "N=${n} server=${server} agg=${agg} steps/s  median=${med}  samples/s=${sam}  peak=${mem} MiB  util=${util}%  failures=${fails}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${n}" "${MPS}" "${server}" "${agg}" "${med}" "${sam}" "${mem}" "${util}" "${fails}" >> "${RESULT}"

  if [ "${fails}" -gt 0 ]; then
    echo "!! ${fails}/${n} workers failed -- see ${WORK}/n${n}/w*.log. Stopping: past here the numbers"
    echo "!! measure failures, not throughput."
    break
  fi
done

echo "=== ${RESULT} ==="
cat "${RESULT}"
