#!/bin/bash
# The EXECUTABLE every snakemake job runs on a CERN HTCondor node.
#
# ⛔️ IT SETS UP pyenv ONLY, NOT THE LCG VIEW. The jobscript snakemake writes RE-INVOKES SNAKEMAKE on
# this node, so the node needs the pyenv 3.14 interpreter snakemake is installed under. Sourcing the
# LCG view here would set PYTHONHOME/PYTHONPATH for ITS python3.13 and kill that interpreter outright
# ("Fatal Python error: Failed to import encodings module"). The detopt environment is established
# separately, inside each rule's own shell, by `scripts/cern/lcg_env.sh`.
#
# ⛔️ CONDOR JOBS INHERIT NOTHING -- no `getenv` is set in the submit file -- so pyenv must be set up
# here explicitly rather than relied upon from .bashrc, which a non-interactive job never sources.
set -e
export PYENV_ROOT=/afs/cern.ch/work/m/maborisy/pyenv
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
cd /afs/cern.ch/work/m/maborisy/detector-linear || exit 90
exec bash "$1"
