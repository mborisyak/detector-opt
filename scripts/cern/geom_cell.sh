#!/bin/bash
# One CAMPAIGN CELL: a full `scripts/bo.py` run at one (config, seed, arm, reveal), on a CERN
# HTCondor GPU node.  Reconstructed from the verified `ship-adapt-sub/adapt_cell.sh`; every export
# below is load-bearing and documented in docs/lxplus-htcondor-gpu.md.
#
# NO `set -u`: the LCG view's setup.sh dereferences unset variables and would exit 1 before anything
# runs.
#
# SCRATCH IS THE WORKING DIRECTORY, AFS IS THE MIRROR.  Node AFS tokens expire ~24 h out and live AFS
# I/O through a long run is the failure mode that doc warns about, so the run writes to
# $_CONDOR_SCRATCH_DIR and rsyncs to AFS every 120 s and on SIGTERM.
#
# STALE MARKERS ARE CLEARED, THE TRAJECTORY IS NOT.  A resubmission must not inherit a previous
# attempt's status.txt -- one was misread as live state on 2026-08-21 -- but `bo.py` RESUMES from
# results.json plus optimizer.npz / trainer.npz, so those are pulled back into scratch instead of
# being wiped.  A cell that ran out of +MaxRuntime therefore CONTINUES when resubmitted.
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh 2>/dev/null
# venv site-packages must PRECEDE the CVMFS one: omegaconf 2.3.1 needs antlr4 4.9.3 and the LCG
# view ships 4.13.1, which raises "Could not deserialize ATN with version (expected 4)".
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd $D/detector-opt || exit 90

CONFIG=$1
SEED=$2
ARM=$3
REVEAL=$4
AFSOUT=$5
PY=$D/lcgvenv/bin/python
SCRATCH=${_CONDOR_SCRATCH_DIR:-/tmp}
WORK=$SCRATCH/cell
mkdir -p "$WORK" "$AFSOUT" || exit 91

rm -f "$AFSOUT/status.txt" "$AFSOUT/CAPPED.txt" "$AFSOUT/preflight.txt"
rsync -a --exclude 'status.txt' --exclude 'CAPPED.txt' --exclude 'preflight.txt' \
      "$AFSOUT/" "$WORK/" 2>/dev/null

echo "host:    $(hostname)"
echo "gpu:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1)"
echo "cell:    config=$CONFIG seed=$SEED arm=$ARM reveal=$REVEAL"
echo "resume:  $([ -f "$WORK/results.json" ] && echo yes || echo no)"
echo "started: $(date -u)"

# Refuse before touching a GPU if the cell is mis-specified, so a bad submission costs nothing.
# The COLUMN-DROP assertion is the point of `stereo_address_design`: withholding must remove only
# the trailing design columns and leave the measurement block byte-identical, or the revealed and
# withheld arms are not comparable and the campaign is meaningless.
$PY - "$CONFIG" "$REVEAL" > "$AFSOUT/preflight.txt" 2>&1 <<'PYEOF'
import sys
import numpy as np, jax, jax.numpy as jnp, yaml
import detopt.detector
from detopt.nn.trainer import REVEAL
config_name, reveal = sys.argv[1], sys.argv[2]
assert reveal in REVEAL, f"{reveal!r} not in {REVEAL}"
cfg = yaml.safe_load(open(f"config/{config_name}.yaml"))
det = detopt.detector.from_config(cfg["detector"])
print("[check] detector          =", list(cfg["detector"])[0], type(det).__name__)
print("[check] reveal            =", reveal)
print("[check] design_dim        =", det.design_dim(), det.design_bounds())
print("[check] features revealed =", det.combined_event_shape(True))
print("[check] features withheld =", det.combined_event_shape(False))
print("[check] this cell will use=", det.combined_event_shape(reveal != "none"))
print("[check] loss_precision    =", cfg["training"]["loss_precision"])
print("[check] budget            =", cfg["training"]["budget"])
u = np.asarray(det.to_scaled(cfg["nominal_design"]), np.float32)
rt = np.asarray(det._to_scaled_flat(det._to_nominal_flat(u)))
print("[check] scaled roundtrip  = %.2e" % float(np.abs(rt - u).max()))
assert np.abs(rt - u).max() < 1e-4, "nominal<->scaled is not a bijection"
pen = det.design_penalty(det.to_nominal(u))
print("[check] design_penalty    =", None if pen is None else float(pen))
b = jax.tree.map(lambda x: jnp.broadcast_to(jnp.asarray(x)[None], (8,) + jnp.asarray(x).shape), det.to_nominal(u))
_g, ev, mask, _t = det(b, np.arange(8, dtype=np.int32))
f = np.asarray(det.combine_scaled(ev, u, mask=mask, reveal_design=(reveal != "none")))
print("[check] combine shape     =", f.shape, " finite =", bool(np.isfinite(f).all()))
print("[check] hits per event    = %.1f" % float((np.asarray(mask) > 0).sum(1).mean()))
assert np.isfinite(f).all(), "combine produced non-finite features"
assert f.shape[-1] == det.combined_event_shape(reveal != "none")[-1], "combine width != combined_event_shape"
g = np.asarray(jax.grad(lambda d: jnp.sum(det.combine_scaled(ev, d, mask=mask) ** 2))(jnp.asarray(u)))
print("[check] d(combine)/d(design) =", g.round(3))
assert np.isfinite(g).all(), "design gradient is not finite"
PYEOF
RC=$?
cat "$AFSOUT/preflight.txt"
if [ $RC -ne 0 ]; then
  echo "status=PREFLIGHT_FAILED" > "$AFSOUT/status.txt"; exit 92
fi

sync_out() { rsync -a "$WORK/" "$AFSOUT/" 2>/dev/null; }
( while true; do sleep 120; sync_out; done ) & SYNCPID=$!
on_term() { sync_out; { echo "status=TRUNCATED"; echo "exit=143"
  echo "reason=SIGTERM (+MaxRuntime); results.json is banked, resubmit to RESUME"; } > "$AFSOUT/status.txt"
  sync_out; exit 143; }
trap on_term TERM INT

set -o pipefail
$PY -u scripts/bo.py "=$CONFIG" output="$WORK" seed="$SEED" \
    nn_init_strategy="$ARM" training.reveal="$REVEAL" 2>&1 | tee -a "$WORK/run.log"
RC=$?
set +o pipefail
kill "$SYNCPID" 2>/dev/null; wait "$SYNCPID" 2>/dev/null; sync_out

STATUS="EXIT_$RC"
if grep -q "did not reach precision within iteration_limit" "$WORK/run.log" 2>/dev/null; then
  STATUS=CAPPED
  grep -h "did not reach precision within iteration_limit" -B 6 "$WORK/run.log" | tail -40 > "$AFSOUT/CAPPED.txt"
  RC=0
elif [ $RC -eq 0 ]; then
  STATUS=COMPLETED
fi
{ echo "status=$STATUS"; echo "exit=$RC"; echo "config=$CONFIG seed=$SEED arm=$ARM reveal=$REVEAL"
  echo "host=$(hostname)"; echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)"
  echo "finished=$(date -u)"; } > "$AFSOUT/status.txt"
sync_out
echo "status=$STATUS exit=$RC"
exit $RC
