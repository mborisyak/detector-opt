#!/bin/bash
# Section-5 measurement of loss_precision. For every [converged/bayes] line, record diff, err,
# diff+err and window; report per-run MEAN and MAXIMUM of the sum against the bar (prec).
# Source: the authoritative repo archive written by scripts/archive_slurm_logs.sh, which
# outlives snakemake deleting the originals on success. Read-only. Usage: margins.sh [full]
ARC=/home/max/dev/detector-opt/output/slurm-log-archive
MODE="${1:-summary}"
shopt -s nullglob
for src in "$ARC"/rule_bo_output_enzyme_extremes_*.log; do
  run=${src##*/rule_bo_output_enzyme_extremes_}
  run=${run%.log}
  awk -v run="$run" -v mode="$MODE" '
    /\[converged\/bayes\]/ {
      diff=""; err=""; win=""; prec=""
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^diff=/)   { sub(/^diff=/, "", $i);   diff = $i + 0 }
        if ($i ~ /^err=/)    { sub(/^err=/, "", $i);    err  = $i + 0 }
        if ($i ~ /^prec=/)   { sub(/^prec=/, "", $i);   prec = $i + 0 }
        if ($i ~ /^window=/) { sub(/^window=/, "", $i); win  = $i + 0 }
      }
      n++; s = diff + err; tot += s
      if (mode == "full") printf "%s  #%d diff=%.4f err=%.4f sum=%.4f window=%d\n", run, n, diff, err, s, win
      if (s > maxs) { maxs = s; maxd = diff; maxe = err; maxw = win; maxi = n }
      p = prec
    }
    END {
      if (n > 0) {
        mean = tot / n
        printf "%s: converged=%d MEANsum=%.4f (%.0f%% of bar) MAXsum=%.4f (%.0f%% of bar; #%d diff=%.4f err=%.4f window=%d) prec=%.4f minmargin=%.4f\n",
               run, n, mean, 100 * mean / p, maxs, 100 * maxs / p, maxi, maxd, maxe, maxw, p, p - maxs
      } else
        printf "%s: converged=0\n", run
    }
  ' "$src"
done
exit 0
