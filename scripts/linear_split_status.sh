#!/bin/bash
# One-screen state of the linear test stage split between the workstation and CERN (2026-09-09), for the 10-minute
# rebalancing tick: per site, the jobs running (with their cells and minutes since start), the queued ones, the
# targets not yet started, and any running job whose cell has not written for more than STALE minutes (default 45).
#
#   scripts/linear_split_status.sh            # both sites
#   STALE=30 scripts/linear_split_status.sh
#
# The target lists live in /tmp/split_local.txt (workstation) and /tmp/split_cern.txt (on the lxplus node the alias
# lands on); a cell counts as started when its directory holds results.json or a job for it is running.
set -u
cd "$(dirname "$0")/.." || exit 90
STALE=${STALE:-45}
NOW=$(date +%s)

echo "=== LOCAL $(date '+%H:%M') ==="
squeue -h -o "%i %T %M %k" 2>/dev/null | sed -E 's/rule_([a-z_]+)_wildcards_output_linear_/\1 /; s/_test_/\//; s/_([0-9]+)_([a-z_]+)$/\/\1\/\2/' \
  | awk '{printf "  %-8s %-9s %-8s %s %s\n", $1, $2, $3, $4, $5}'
echo "  done $(find output/linear -path '*/test/*' -name done.txt | wc -l)/120  verified $(find output/linear -path '*/test/*' -name verified.txt | wc -l)/120 (local tree)"
if [ -f /tmp/split_local.txt ]; then
  pending=0; started=0; finished=0
  while read -r t; do
    [ -z "$t" ] && continue
    cell=${t%/verified.txt}
    if [ -f "$t" ]; then finished=$((finished + 1))
    elif [ -f "$cell/results.json" ] || squeue -h -o "%k" | grep -q "$(echo "$cell" | sed 's#output/linear/##; s#/test/#_test_#; s#/#_#g')"; then started=$((started + 1))
    else pending=$((pending + 1)); fi
  done < /tmp/split_local.txt
  echo "  local targets: $finished finished, $started started, $pending not started"
fi
echo "  stale check (running cells with no write for > $STALE min):"
for spec in $(squeue -h -t R -o "%k|%M" 2>/dev/null); do
  k=${spec%%|*}; m=${spec##*|}
  cell=$(echo "$k" | sed -E 's/rule_[a-z_]+_wildcards_//; s/^output_linear_([a-z0-9]+)_([0-9]+)_([a-z_]+)$/output\/linear\/\1\/test\/\2\/\3/')
  newest=$(find "$cell" -maxdepth 2 -type f -newermt "-$STALE minutes" 2>/dev/null | head -1)
  elapsed_min=$(echo "$m" | awk -F: '{ if (NF==3) print $1*60+$2; else print $1 }')
  if [ -z "$newest" ] && [ "${elapsed_min:-0}" -gt "$STALE" ]; then echo "    STALE: $cell (running $m, nothing written in $STALE min)"; fi
done

echo "=== CERN ==="
timeout 120 ssh -o BatchMode=yes lxplus bash -s "$STALE" <<'EOF'
STALE=$1
cd /afs/cern.ch/work/m/maborisy/detector-opt || exit 1
echo "  node $(hostname); orchestrators: $(pgrep -af 'run_linear_main.sh|run_reveal.sh' | grep -v pgrep | awk '{print $3}' | sort | uniq -c | tr '\n' ' ')"
condor_q -af JobBatchName JobStatus 2>/dev/null | awk '{s[$1" "($2==1?"idle":($2==2?"run":($2==5?"HELD":$2)))]++} END {for (k in s) print "    "k": "s[k]}' | sort
echo "  done $(ls output/linear/*/test/*/*/done.txt | wc -l)/120  verified $(ls output/linear/*/test/*/*/verified.txt | wc -l)/120 (CERN tree)"
if [ -f /tmp/split_cern.txt ]; then
  pending=0; started=0; finished=0
  while read -r t; do
    [ -z "$t" ] && continue; cell=${t%/verified.txt}
    if [ -f "$t" ]; then finished=$((finished + 1))
    elif [ -f "$cell/results.json" ] || condor_q -constraint 'JobStatus == 2' -af SnakeCell 2>/dev/null | grep -q "$cell"; then started=$((started + 1))
    else pending=$((pending + 1)); fi
  done < /tmp/split_cern.txt
  echo "  CERN targets: $finished finished, $started started, $pending not started"
fi
echo "  running jobs (min since start, node, cell) and stale check (> $STALE min without a write):"
now=$(date +%s)
condor_q -constraint 'JobStatus == 2' -af ClusterId JobBatchName JobCurrentStartDate RemoteHost SnakeCell 2>/dev/null | while read -r id batch start host cell; do
  mins=$(( (now - start) / 60 )); dir=${cell%/done}; dir=${dir%/verified}
  if [ "$dir" = "unknown" ] && [ "$batch" = "reveal-reveal_bo" ]; then dir=$(ls -d output/ablation-reveal-test/*/*/continue 2>/dev/null | while read -r c; do test -f "$c/verified.txt" || echo "$c"; done | head -1); fi
  newest=$(find "$dir" -maxdepth 2 -type f -newermt "-$STALE minutes" 2>/dev/null | head -1)
  flag=""; if [ -z "$newest" ] && [ "$mins" -gt "$STALE" ]; then flag="  <-- STALE"; fi
  printf "    %s %-14s %4d min  %-28s %s%s\n" "$id" "$batch" "$mins" "${host#*@}" "${dir#output/}" "$flag"
done
echo "  on b9pgpun102: $(condor_q -constraint 'regexp("b9pgpun102", RemoteHost)' -af ClusterId 2>/dev/null | wc -l)"
EOF
