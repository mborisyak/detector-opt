#!/bin/bash
# PROMOTE `bo2` after `bo` has died, and restart the intersect campaign there.
#
#   scripts/migrate_bo_to_bo2.sh            # check only, change nothing
#   scripts/migrate_bo_to_bo2.sh --go       # actually migrate
#
# WHY THIS EXISTS. The two boxes run as ONE cluster with `bo` as both SLURM controller and NFS server,
# so `bo` dying takes `bo2`'s filesystem with it -- the price of a single cluster over two independent
# sites. That is survivable only because the workstation holds a full mirror refreshed every 5 minutes
# by `scripts/evacuate_bo_loop.sh`: results, checkpoints and all. This script turns that mirror back
# into a running campaign on `bo2`.
#
# ⛔️ CHECKPOINTS ARE NOT OPTIONAL. `verify_trajectory.py` restores the network each design was REPORTED
# with from `checkpoints/design_NNNN`. A results-only restore looks complete -- done.txt present,
# designs banked -- and then fails hours later when verification reaches it. That happened once already
# and cost two recomputed cells. This script copies the WHOLE tree.
#
# WHAT IT DOES NOT DO. It does not resurrect in-flight cells: whatever `bo` was computing when it died
# is lost back to its last banked design, which `bo.py` resumes from. Expect to lose up to one design
# per running cell, plus anything written in the last 5 minutes.
set -u
GO=0
[ "${1:-}" = "--go" ] && GO=1
cd "$(dirname "$0")/.." || exit 90
MIRROR=output/final-bo

say() { printf '  %s\n' "$*"; }

say "=== 1. is bo actually down? ==="
if timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=15 bo 'echo alive' >/dev/null 2>&1; then
  say "bo is REACHABLE -- refusing to migrate. Stop it deliberately first if that is what you want."
  exit 1
fi
say "bo unreachable, as expected for a migration"

say "=== 2. is bo2 ready to take over? ==="
READY=$(timeout 40 ssh -o BatchMode=yes bo2 '
  printf "%s %s %s %s" \
    "$([ -x /root/venv/bin/python ] && echo venv || echo NOVENV)" \
    "$(command -v sinfo >/dev/null && echo slurm || echo NOSLURM)" \
    "$([ -f /root/detector-opt/detopt/detector/straw_detector.so ] && echo detector || echo NODETECTOR)" \
    "$([ -d /root/detector-opt/data/mc/numpy_newFS ] && echo data || echo NODATA)"' 2>/dev/null)
say "bo2 reports: ${READY:-unreachable}"
case "$READY" in
  *NOVENV*|*NOSLURM*|*NODETECTOR*|"") say "bo2 is NOT ready -- provision it before migrating"; exit 1;;
esac

say "=== 3. what the mirror holds ==="
say "cells=$(find "$MIRROR" -name results.json 2>/dev/null | wc -l) done=$(find "$MIRROR" -name done.txt 2>/dev/null | wc -l) verified=$(find "$MIRROR" -name verified.txt 2>/dev/null | wc -l) checkpoints=$(find "$MIRROR" -name manifest.ocdbt 2>/dev/null | wc -l) size=$(du -sh "$MIRROR" 2>/dev/null | cut -f1)"
say "mirror last refreshed: $(tail -1 logs/evacuate-bo.log 2>/dev/null)"

if [ "$GO" -eq 0 ]; then
  say ""
  say "CHECK ONLY -- nothing changed. Re-run with --go to migrate."
  exit 0
fi

say "=== 4. make bo2 a standalone controller (it was a compute node of bo's cluster) ==="
timeout 120 ssh -o BatchMode=yes bo2 'set -u
  NEW=$(hostname)
  # Drop the dead controller and the second node; bo2 becomes controller AND the only node.
  sed -i "s/^SlurmctldHost=.*/SlurmctldHost=$NEW/" /etc/slurm/slurm.conf
  grep -v "^NodeName=" /etc/slurm/slurm.conf > /tmp/sc && mv /tmp/sc /etc/slurm/slurm.conf
  REAL=$(slurmd -C | grep -oE "RealMemory=[0-9]+" | cut -d= -f2)
  echo "NodeName=$NEW CPUs=60 Boards=1 SocketsPerBoard=60 CoresPerSocket=1 ThreadsPerCore=1 RealMemory=$((REAL-12000)) Gres=gpu:2,shard:12 State=UNKNOWN" >> /etc/slurm/slurm.conf
  grep -q "^PartitionName=" /etc/slurm/slurm.conf || echo "PartitionName=main Nodes=ALL Default=YES MaxTime=INFINITE State=UP" >> /etc/slurm/slurm.conf
  grep -v "$(hostname)" /etc/slurm/gres.conf | grep -q . && sed -i "/^NodeName=/!d" /etc/slurm/gres.conf
  awk -v h="$NEW" "/^NodeName=/ && \$0 !~ h {next} {print}" /etc/slurm/gres.conf > /tmp/gc && mv /tmp/gc /etc/slurm/gres.conf
  umount /root/detector-opt 2>/dev/null   # the dead NFS mount, if it was up
  systemctl enable --now slurmctld >/dev/null 2>&1
  systemctl restart munge; sleep 2; systemctl restart slurmctld; sleep 3; systemctl restart slurmd; sleep 4
  scontrol update NodeName=$NEW State=RESUME >/dev/null 2>&1; sleep 2
  echo "  node: $(sinfo -h -o "%N %t") gres: $(scontrol show node | grep -oE "Gres=[^ ]+")"'

say "=== 5. restore the tree from the mirror (checkpoints included) ==="
rsync -a "$MIRROR/" bo2:/root/detector-opt/output/
timeout 60 ssh -o BatchMode=yes bo2 'cd /root/detector-opt && echo "  restored: intersect $(ls output/intersect/select/*/*/*/done.txt 2>/dev/null|wc -l)/45 done, $(ls output/intersect/select/*/*/*/verified.txt 2>/dev/null|wc -l)/45 verified, $(find output -name manifest.ocdbt 2>/dev/null|wc -l) checkpoints"'

say "=== 6. relaunch intersect on bo2 ==="
timeout 120 ssh -o BatchMode=yes bo2 'cd /root/detector-opt || exit 1
  export PATH=/root/venv/bin:$PATH
  snakemake -s ship.snake --unlock >/dev/null 2>&1
  setsid nohup flock -n /tmp/ship-bo.lock \
    snakemake -s ship.snake --profile profiles/bo --rerun-incomplete output/intersect/campaign.txt \
    > logs/ship-bo.log 2>&1 < /dev/null &
  sleep 40
  echo "  orchestrator=$(pgrep -f "bin/snake""make"|wc -l) squeue=$(squeue -h|wc -l) errors=$(grep -cE "^Error in rule" logs/ship-bo.log 2>/dev/null)"'

say "=== 7. REPOINT THE EVACUATION LOOP -- it is still syncing from the dead host ==="
say "run:  pkill -f evacuate_bo_loop.sh   then edit SRC to bo2 and relaunch, or the mirror goes stale"
