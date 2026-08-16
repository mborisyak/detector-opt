#!/usr/bin/env bash
# Verification throughput against PROCESS COUNT, under CUDA MPS.
#
# Runs N concurrent `verify_trajectory.py` processes at a ~1 GB dataset (config/_profile_verify.yaml,
# verify.budget 1572864 = 0.94 GB at 600 B/event) and reports per-process wall time and aggregate
# throughput for each N. Without MPS, N processes TIME-SLICE one GPU and aggregate throughput is flat;
# with MPS they co-run and it should rise until memory or SMs saturate. That difference is what this
# measures.
#
# ONE SLURM JOB owns the whole GPU (`--gres=shard:2`) and fans out internally. Submitting N separate
# GPU jobs would be capped at the 2 shards the node advertises, and launching them outside SLURM would
# make its accounting a lie.
#
#   sbatch --gres=shard:2 --cpus-per-task=10 --mem=20000 -J vprof -o logs/profile-verify.log \
#          --wrap='scripts/profile_verify_parallel.sh "1 2 4 6 8"'
set -u

LADDER=${1:-"1 2 4 8"}
# MPS=on routes every client through the control daemon's pipe; MPS=off leaves it unset so each process
# creates its own context. Under Default compute mode nothing forces the choice, so a client that does
# not set the pipe directory silently TIME-SLICES instead of co-running -- which is the whole thing
# being measured, and is easy to report as MPS by mistake.
MPS=${MPS:-on}
MPS_PIPE=${MPS_PIPE:-$HOME/.mps}
CONFIG=${CONFIG:-=_profile_verify}
TRAJECTORY=${TRAJECTORY:-output/enzyme_extremes/126382657/from_scratch/results.json}
WORK=${WORK:-/tmp/claude-1000/-home-max-dev-detector-opt/f322215c-c143-48c0-ad31-e515686eef05/scratchpad/vprof}
RESULT=${RESULT:-output/profile-verify-${MPS:-on}.tsv}

mkdir -p "${WORK}"
printf 'n_procs\trep\twall_s\tmed_wall_s\tthroughput_per_min\tgpu_mem_mib\tgpu_util_pct\n' > "${RESULT}"

if [ "${MPS}" = "on" ]; then
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE}"
  echo "MPS: ON, pipe ${CUDA_MPS_PIPE_DIRECTORY}"
else
  unset CUDA_MPS_PIPE_DIRECTORY
  echo "MPS: OFF (control daemon may run, but clients will not connect)"
fi
echo "MPS control daemon: $(pgrep -f '[n]vidia-cuda-mps-control' >/dev/null && echo RUNNING || echo ABSENT)"
nvidia-smi --query-gpu=compute_mode,memory.total --format=csv,noheader

for n in ${LADDER}; do
  echo "=================== N = ${n} ==================="
  rm -rf "${WORK}/n${n}"; mkdir -p "${WORK}/n${n}"

  # Sample the card while the batch runs: peak memory and mean utilisation are what say whether the
  # processes co-ran or queued behind each other.
  ( while true; do
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits
      sleep 2
    done ) > "${WORK}/n${n}/gpu.csv" 2>/dev/null &
  sampler=$!

  start=$(date +%s.%N)
  # Collect the WORKER pids and wait on those alone. A bare `wait` also waits for the sampler, which
  # is a `while true` loop that never exits -- the batch then hangs after every worker has finished.
  pids=()
  for i in $(seq 1 "${n}"); do
    ( t0=$(date +%s.%N)
      python -u scripts/verify_trajectory.py "${CONFIG}" \
        trajectory="${TRAJECTORY}" \
        output="${WORK}/n${n}/proc${i}.json" \
        seed=$((1000 + i)) progress=none force=yes > "${WORK}/n${n}/proc${i}.log" 2>&1
      t1=$(date +%s.%N)
      echo "${i} $(echo "${t1} - ${t0}" | bc) $?" > "${WORK}/n${n}/proc${i}.time"
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  end=$(date +%s.%N)
  kill "${sampler}" 2>/dev/null
  wait "${sampler}" 2>/dev/null

  batch=$(echo "${end} - ${start}" | bc)
  times=$(cat "${WORK}/n${n}"/proc*.time 2>/dev/null | awk '{print $2}' | sort -g)
  med=$(echo "${times}" | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  mem=$(awk -F, 'BEGIN{m=0} {gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END{print m}' "${WORK}/n${n}/gpu.csv")
  util=$(awk -F, '{gsub(/ /,"",$2); s+=$2; c++} END{if(c) printf "%.0f", s/c}' "${WORK}/n${n}/gpu.csv")
  thr=$(echo "scale=3; ${n} * 60 / ${batch}" | bc)
  fails=$(cat "${WORK}/n${n}"/proc*.time 2>/dev/null | awk '$3!=0' | wc -l)

  server=$(pgrep -f "[n]vidia-cuda-mps-server" >/dev/null && echo yes || echo no)
  echo "N=${n}  mps_server=${server}  batch ${batch}s  median/proc ${med}s  throughput ${thr}/min  peak ${mem} MiB  util ${util}%  failures ${fails}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${n}" "1" "${batch}" "${med}" "${thr}" "${mem}" "${util}" >> "${RESULT}"

  if [ "${fails}" -gt 0 ]; then
    echo "!! ${fails} of ${n} processes exited non-zero at N=${n} -- see ${WORK}/n${n}/proc*.log"
    echo "!! stopping the ladder: past this point the numbers measure failures, not throughput"
    break
  fi
done

echo "=== results -> ${RESULT} ==="
cat "${RESULT}"
