#!/usr/bin/env bash
# Provision the preemptible cloud box `bo` to MIRROR this workstation's environment.
#
# Idempotent and re-runnable: on a preemptible instance the recovery path must be one command, not a
# memory of the session that first ran it. Every stage checks for its own result and skips.
#
# MIRRORING, not "a working environment". The numerics are decided by CPython 3.11.9 plus the exact
# package pins in requirements-bo.txt, taken from `pip freeze` on the workstation. jax 0.10.2 with
# the cuda13 plugin is also what the Blackwell card needs (sm_120 requires CUDA 13), so the pinned
# stack is the correct stack here rather than a compromise. `gearup` is an editable git checkout on
# the workstation, not a PyPI release, so it is rsynced separately and installed -e; taking the PyPI
# release instead would be a silent divergence.
#
# Installs are authorised ON THIS REMOTE BOX ONLY (docs/cloud-campaign-runbook.md section 0, and the
# user's explicit request plus passwordless sudo). The workstation is never installed to.
#
# Run ON bo:  bash ~/detector-opt/scripts/provision_bo.sh
set -euo pipefail

PYTHON_VERSION=3.11.9
PYENV_ROOT="$HOME/.pyenv"
VENV="$HOME/venv"
REPO="$HOME/detector-opt"
GEARUP="$HOME/gearup"

log() { printf '\n=== %s ===\n' "$*"; }

log "apt build dependencies"
if ! dpkg -s libssl-dev >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
    libffi-dev liblzma-dev tk-dev libncursesw5-dev xz-utils curl git
else
  echo "already present"
fi

log "pyenv"
if [ ! -d "$PYENV_ROOT" ]; then
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

log "acceptance: jax on GPU"
"$VENV/bin/python" - <<'PY'
import jax
backend = jax.default_backend()
print("jax", jax.__version__, "backend", backend, jax.devices())
assert backend == "gpu", f"jax is on {backend}, not gpu -- STOP, do not launch the campaign"
x = jax.numpy.ones((4096, 4096))
print("matmul ok", float((x @ x).sum()))
PY

log "done"
echo "activate with: source $VENV/bin/activate"
