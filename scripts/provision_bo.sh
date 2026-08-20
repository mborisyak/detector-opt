#!/usr/bin/env bash
# Provision a preemptible cloud box to MIRROR this workstation's environment, then give it SLURM and
# MPS so campaigns can be scheduled on it.
#
# Idempotent and re-runnable: on a preemptible instance the recovery path must be one command, not a
# memory of the session that first ran it. Every stage checks for its own result and skips. One box
# was reclaimed mid-provision already; that must cost minutes, not a rebuild.
#
# THE WORK DISK IS THE POINT. `WORK` defaults to a separate volume that SURVIVES the machine being
# stopped, and pyenv, the venv, the repo and all output live on it. What does NOT survive is the OS
# root: apt packages, the SLURM config, fstab and the MPS daemon. So a reclaimed box re-runs this
# script, the expensive stages (CPython build, ~10 GB of CUDA wheels, the data) find their results
# already on the disk and skip, and only the cheap OS-level stages actually run.
#
# MIRRORING, not "a working environment". The numerics are decided by CPython 3.11.9 plus the exact
# package pins in requirements-bo.txt, taken from `pip freeze` on the workstation. jax 0.10.2 with
# the cuda13 plugin is also what the Blackwell cards need (sm_120 requires CUDA 13), so the pinned
# stack is the correct stack here rather than a compromise. `gearup` is an editable git checkout on
# the workstation, not a PyPI release, so it is rsynced separately and installed -e; taking the PyPI
# release instead would be a silent divergence.
#
# THE STRAW DETECTOR MUST BE REBUILT HERE and cannot be shipped. The Makefile compiles it with
# `-march=native` and links the interpreter's own libpython, so a `.so` copied from the workstation
# is wrong twice over. `make` runs against the venv's python, after the venv exists.
#
# SHARDS, NOT GPUS. Each card is exposed as SHARDS_PER_GPU shards so several processes share it
# through MPS, and each task requests the count that matches ITS measured scaling: SHiP takes 3 of 6
# (two processes per card, its measured 1.4x), while a task that scales to 6x on 12 processes takes
# 1. Requesting `--gres=gpu:1` instead would take a whole card exclusively and collide with the
# shard jobs on it. CPUs are requested ONE per job: most scripts are single-core, so the card and
# not the core count is what bounds concurrency.
#
# NO cgroup.conf IS WRITTEN. `TaskPlugin=task/none` plus `ProctrackType=proctrack/linuxproc` need no
# cgroup constraints, and slurmd loads the cgroup plugin at init whenever the file exists -- with an
# invalid `CgroupPlugin` it then refuses to start, which is how this box first failed.
#
# SLURM here is single-node with NO slurmdbd, exactly like the workstation, so `sacct` never answers
# and snakemake needs `--slurm-status-command squeue`.
#
# Installs are authorised ON THE REMOTE BOX ONLY (the user's explicit request, plus root). The
# workstation is never installed to.
#
# Run ON the box:  bash /mnt/work/repo/scripts/provision_bo.sh
set -euo pipefail

PYTHON_VERSION=3.11.9
WORK=${WORK:-/mnt/work}
WORK_LABEL=${WORK_LABEL:-work}
PYENV_ROOT=${PYENV_ROOT:-$WORK/pyenv}
VENV=${VENV:-$WORK/venv}
REPO=${REPO:-$WORK/repo}
GEARUP=${GEARUP:-$WORK/gearup}
DATA=${DATA:-$WORK/data}
DATA_LINK=${DATA_LINK:-/home/max/dev/data}
SHARDS_PER_GPU=${SHARDS_PER_GPU:-6}

if [ "$(id -u)" -eq 0 ]; then SUDO="env"; else SUDO="sudo env"; fi

log() { printf '\n=== %s ===\n' "$*"; }

log "work disk"
if ! mountpoint -q "$WORK"; then
  mkdir -p "$WORK"
  if blkid -L "$WORK_LABEL" >/dev/null 2>&1; then
    $SUDO mount "$(blkid -L "$WORK_LABEL")" "$WORK"
  else
    echo "no volume labelled $WORK_LABEL -- using $WORK on the root filesystem" >&2
  fi
fi
grep -q "$WORK" /etc/fstab 2>/dev/null || echo "LABEL=$WORK_LABEL $WORK ext4 defaults,nofail 0 2" | $SUDO tee -a /etc/fstab >/dev/null
df -h "$WORK" | tail -1

log "apt build dependencies"
if ! dpkg -s libssl-dev >/dev/null 2>&1; then
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get update -qq
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev tk-dev libncursesw5-dev xz-utils curl git rsync
else
  echo "already present"
fi

log "data symlink ($DATA_LINK -> $DATA)"
mkdir -p "$(dirname "$DATA_LINK")"
[ -e "$DATA_LINK" ] || ln -s "$DATA" "$DATA_LINK"
ls "$DATA_LINK" 2>/dev/null | head -5

log "pyenv"
if [ ! -x "$PYENV_ROOT/bin/pyenv" ]; then
  rm -rf "$PYENV_ROOT"
  git clone --depth 1 https://github.com/pyenv/pyenv.git "$PYENV_ROOT"
else
  echo "already present"
fi
export PYENV_ROOT
export PATH="$PYENV_ROOT/bin:$PATH"

log "CPython $PYTHON_VERSION"
if [ ! -x "$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python" ]; then
  MAKE_OPTS="-j$(nproc)" pyenv install "$PYTHON_VERSION"
else
  echo "already built"
fi

log "venv"
if [ ! -x "$VENV/bin/python" ]; then
  "$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip setuptools wheel

log "pinned packages"
"$VENV/bin/python" -m pip install --quiet -r "$REPO/requirements-bo.txt"

log "gearup (editable, mirrors the workstation checkout)"
if [ -d "$GEARUP" ]; then
  "$VENV/bin/python" -m pip install --quiet --no-deps -e "$GEARUP"
else
  echo "MISSING $GEARUP -- rsync it from the workstation first" >&2
  exit 1
fi

log "detopt (editable)"
"$VENV/bin/python" -m pip install --quiet --no-deps -e "$REPO"

log "straw detector (built here, -march=native)"
make -C "$REPO" PYTHON="$VENV/bin/python" all
"$VENV/bin/python" -c "import detopt.detector.straw as s; print('straw ok', s.__file__)"

log "slurm + munge"
HOST=$(hostname)
N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
N_SHARD=$((N_GPU * SHARDS_PER_GPU))
if ! command -v sinfo >/dev/null 2>&1; then
  $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq slurm-wlm munge
fi
$SUDO test -f /etc/munge/munge.key || $SUDO /usr/sbin/mungekey --create --keyfile /etc/munge/munge.key
$SUDO chown munge:munge /etc/munge/munge.key
$SUDO chmod 400 /etc/munge/munge.key
$SUDO mkdir -p /var/spool/slurmctld /var/spool/slurmd /var/log/slurm
$SUDO chown slurm:slurm /var/spool/slurmctld /var/log/slurm
$SUDO chmod 755 /var/spool/slurmctld /var/spool/slurmd
$SUDO tee /etc/slurm/slurm.conf >/dev/null <<CONF
ClusterName=bo
SlurmctldHost=${HOST}
SlurmUser=slurm
SlurmdUser=root
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
AuthType=auth/munge
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu,shard
AccountingStorageType=accounting_storage/none
JobAcctGatherType=jobacct_gather/none
ReturnToService=2
MaxJobCount=10000
NodeName=${HOST} CPUs=$(nproc) RealMemory=$(($(free -m | awk '/^Mem:/{print $2}') - 16000)) Gres=gpu:${N_GPU},shard:${N_SHARD} State=UNKNOWN
PartitionName=main Nodes=ALL Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO
CONF
{
  echo "AutoDetect=off"
  for i in $(seq 0 $((N_GPU - 1))); do echo "Name=gpu File=/dev/nvidia${i}"; done
  for i in $(seq 0 $((N_GPU - 1))); do echo "Name=shard Count=${SHARDS_PER_GPU} File=/dev/nvidia${i}"; done
} | $SUDO tee /etc/slurm/gres.conf >/dev/null
$SUDO rm -f /etc/slurm/cgroup.conf
$SUDO systemctl enable munge slurmctld slurmd >/dev/null 2>&1 || true
$SUDO systemctl restart munge
sleep 1
$SUDO systemctl restart slurmctld slurmd
sleep 4
sinfo -o "%P %a %D %t %C %G"

log "mps"
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
chmod 1777 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
for i in $(seq 0 $((N_GPU - 1))); do
  nvidia-smi -i "$i" -c DEFAULT >/dev/null
done
if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
  $SUDO CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((N_GPU - 1)))" \
        CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY" \
        CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_LOG_DIRECTORY" \
        nvidia-cuda-mps-control -d
  sleep 2
fi
echo get_server_list | nvidia-cuda-mps-control 2>&1 | head -5
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 && echo "mps control daemon up" || { echo "MPS FAILED TO START" >&2; exit 1; }
$SUDO tee /etc/profile.d/mps.sh >/dev/null <<RC
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
RC
$SUDO tee "$WORK/mps-env.sh" >/dev/null <<'RC'
# Source inside a SLURM job wrap to become an MPS client sized to the shards the job was given.
#
# CUDA_MPS_ACTIVE_THREAD_PERCENTAGE caps the SMs one client may take, and without it every client
# asks for the whole card and MPS time-slices them -- which is the flat-to-decreasing regime. The
# cap is the job's shard share: SHiP asks for 3 of 6 shards on a card and so gets 50%.
SHARDS_PER_GPU=${SHARDS_PER_GPU:-6}
granted=${SLURM_GRES:-${SLURM_JOB_GRES:-shard:1}}
granted=${granted##*shard:}
granted=${granted%%,*}
case "$granted" in ''|*[!0-9]*) granted=1 ;; esac
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$((100 * granted / SHARDS_PER_GPU))
RC

log "acceptance: jax on GPU"
"$VENV/bin/python" - <<'PY'
import jax
backend = jax.default_backend()
print("jax", jax.__version__, "backend", backend, jax.devices())
assert backend == "gpu", f"jax is on {backend}, not gpu -- STOP, do not launch the campaign"
x = jax.numpy.ones((4096, 4096))
print("matmul ok", float((x @ x).sum()))
PY

log "acceptance: slurm places a shard job"
srun --gres=shard:1 --cpus-per-task=1 --time=00:02:00 -u hostname

log "done"
echo "activate with: source $VENV/bin/activate"
