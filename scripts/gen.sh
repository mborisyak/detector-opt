#!/bin/bash

ROOT="${1}"
ENERGY="${2}"
SEED="${3}"
NUM="${4}"
DESIGN_START="${5}"
DESIGN_END="${6}"

# export ALIBUILD_WORK_DIR="/afs/cern.ch/work/m/maborisy"
# cd "/afs/cern.ch/work/m/maborisy"
# source /cvmfs/ship.cern.ch/25.01/setUp.sh/
# eval $(alienv load FairShip/latest-sst-optimisation-release --no-refresh)
export PYTHONPATH="$PYTHONPATH:/afs/cern.ch/work/m/maborisy/FairShip/python/"

for i in $(seq ${DESIGN_START} ${DESIGN_END}); do  
  OUTPUT="$ROOT/gen-${ENERGY}-${SEED}-${i}"
  mkdir -p "$OUTPUT"
  # alienv enter FairShip/latest-sst-optimisation-release
  if [ ! -f "$OUTPUT/ship.conical.Pythia8-TGeant4.root" ]; then
      python /afs/cern.ch/work/m/maborisy/FairShip/macro/run_simScript.py -m "$ENERGY" -n "$NUM" -o "$OUTPUT" -s "$SEED" --strawtubes-yaml=/afs/cern.ch/work/m/maborisy/designs/random/strawtubes_${i}.yaml
  else
      echo "Skipping $OUTPUT"
  fi
  if [ ! -f "$OUTPUT/ship.conical.Pythia8-TGeant4_rec.root" ]; then
      cd "$OUTPUT"
      python /afs/cern.ch/work/m/maborisy/FairShip/macro/ShipReco.py -f ship.conical.Pythia8-TGeant4.root -g geofile_full.conical.Pythia8-TGeant4.root
  else
      echo skipping "reco for $OUTPUT"
  fi
  ((SEED++))
done
