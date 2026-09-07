#!/bin/bash
# Poll the fresh SHiP campaign at CERN every 15 min; print a line on any change; flag held jobs, errors, a dead orchestrator.
R=/afs/cern.ch/work/m/maborisy/detector-opt
prev=""
while true; do
  c=$(ssh -o BatchMode=yes -o ConnectTimeout=30 lxplus "cd $R && q=\$(condor_q maborisy -af JobStatus 2>/dev/null | sort | uniq -c | awk '{printf \"%s:%s \", \$2, \$1}'); d=\$(ls output/*/select/*/*/*/done.txt 2>/dev/null | wc -l); v=\$(ls output/*/select/*/*/*/verified.txt 2>/dev/null | wc -l); t=\$(ls output/*/test/*/*/done.txt 2>/dev/null | wc -l); e=\$(grep -cE '^Error|Traceback|RESOURCE_EXHAUSTED' logs/campaign.log 2>/dev/null); o=\$(pgrep -f run_campaign.sh | wc -l); echo \"queue[status:count]=\$q done=\$d/90 verified=\$v/90 test_done=\$t/60 log_errors=\$e orchestrator=\$o\"" 2>/dev/null)
  if [ -n "$c" ] && [ "$c" != "$prev" ]; then
    echo "CAMPAIGN: $c"
    prev="$c"
    case "$c" in *"5:"*|*"orchestrator=0"*) echo "ATTENTION: held jobs or orchestrator gone";; esac
  fi
  sleep 900
done
