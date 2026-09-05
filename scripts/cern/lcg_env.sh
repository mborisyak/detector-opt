# SOURCEABLE environment for detopt work on a CERN HTCondor node. Sourced INSIDE a rule's own shell,
# never in the job wrapper: the wrapper's process runs snakemake under pyenv 3.14, and the LCG view
# sets PYTHONHOME/PYTHONPATH for its own python3.13, which leaves the 3.14 interpreter unable to find
# its standard library ("Fatal Python error: Failed to import encodings module"). Keeping the two in
# separate processes is the whole point -- a rule's shell is a child, so this pollutes nothing.
#
# ⛔️ `set +u` IS LOAD-BEARING AND MUST COME FIRST. snakemake runs every rule's shell with
# `set -euo pipefail`, and the LCG view's setup.sh dereferences unset variables -- under `set -u`
# that aborts it PARTWAY, leaving a PATH with no python, and the rule dies with exit 127 having
# printed nothing. This is the gotcha `geom_cell.sh` documents; it applies here because snakemake
# imposes the strict mode, not because the script asked for it.
#
# The stderr of setup.sh is NOT suppressed: hiding it is what made the first failure silent.
set +u
export CUDA_VISIBLE_DEVICES=${_CONDOR_AssignedGPUs:-}
V=/cvmfs/sft.cern.ch/lcg/views/LCG_109_cuda/x86_64-el9-gcc13-opt
D=/afs/cern.ch/work/m/maborisy
source $V/setup.sh
# The venv site-packages must PRECEDE the CVMFS one: omegaconf 2.3.1 needs antlr4 4.9.3 and the LCG
# view ships 4.13.1, which raises "Could not deserialize ATN with version (expected 4)".
export PYTHONPATH=$D/detector-opt:$D/lcgvenv/lib/python3.13/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=$D/lcgvenv/lib/python3.13/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# A cheap, loud check that the view actually landed -- better than discovering it as exit 127.
command -v python >/dev/null || { echo "lcg_env.sh: no python on PATH after sourcing $V/setup.sh" >&2; return 1 2>/dev/null || exit 1; }
