"""What the arm ratio WOULD have been at another `loss_precision`, from the campaign's own margins.

A design stops when its train/validation gap `diff` plus its statistical error `err` fits under
`loss_precision`. `err` falls as c/sqrt(window); `diff` is an overfitting BIAS floor and need not
fall at all (`scripts/probe_precision.py` states this and the campaign logs bear it out). So from the
one (window, diff, err) triple each design records at convergence we can recover its noise constant

    c = err * sqrt(window)

and read off the window it would have needed against a different bar P:

    window(P) = (c / (P - diff))**2      undefined when P <= diff -- the design CANNOT converge

This is a counterfactual, not a measurement, and it rests on ONE assumption which this script also
tests: that `diff` does not depend on the bar. `loss_precision` additionally enters `is_plateaued`
(via `flatness_tol * loss_precision`) and the large-gap rule, so a real run at another bar follows a
different trajectory; the GPU sweep in `scripts/probe_meta_capacity.py --loss-precision` is the
measurement, and this is the prediction it is checked against.

Usage: python scripts/probe_meta_counterfactual.py <label>=<log> [<label>=<log> ...]
"""

from __future__ import annotations

import math
import statistics as st
import sys

from probe_meta_decompose import parse

BARS = [0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.014, 0.020]


def designs(path):
  return [x for x in parse(path) if x.get("diff") is not None and x.get("window") is not None]


def cost_at(rows, bar):
  """Median counterfactual window at `bar`, and how many designs could never reach it."""
  ok, dead = [], 0
  for x in rows:
    head = bar - x["diff"]
    if head <= 0.0:
      dead += 1
      continue
    ok.append((x["err"] * math.sqrt(x["window"]) / head)**2)
  return (st.median(ok) if len(ok) > 0 else float("nan")), dead, len(rows)


def gap_vs_window(rows):
  """Spearman-ish check that `diff` is a floor: correlation of diff with window across designs."""
  if len(rows) < 4:
    return float("nan")
  w = [x["window"] for x in rows]
  d = [x["diff"] for x in rows]
  rw = {v: i for i, v in enumerate(sorted(set(w)))}
  rd = {v: i for i, v in enumerate(sorted(set(d)))}
  a = [rw[v] for v in w]
  b = [rd[v] for v in d]
  ma, mb = st.mean(a), st.mean(b)
  num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
  den = math.sqrt(sum((x - ma)**2 for x in a) * sum((y - mb)**2 for y in b))
  return num / den if den > 0 else float("nan")


if __name__ == "__main__":
  rows = {}
  for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    rows[label] = designs(path)
    r = rows[label]
    print(
      f"{label:26s} n={len(r):3d}  diff med={st.median([x['diff'] for x in r]):.5f} "
      f"max={max(x['diff'] for x in r):.5f}  c med={st.median([x['err']*math.sqrt(x['window']) for x in r]):.3f}  "
      f"rank-corr(diff, window)={gap_vs_window(r):+.2f}"
    )

  print("\nCOUNTERFACTUAL median window, and (designs that could NEVER converge at that bar)")
  header = "".join(f"{b:>13.4f}" for b in BARS)
  print(f"{'arm':26s}{header}")
  table = {}
  for label, r in rows.items():
    cells, line = [], f"{label:26s}"
    for b in BARS:
      med, dead, n = cost_at(r, b)
      cells.append(med)
      line += f"{med:>9.0f}({dead:>1d})" if med == med else f"{'--':>13s}"
    table[label] = cells
    print(line)

  print("\nCOUNTERFACTUAL RATIO from_scratch / meta versus the bar")
  print(f"{'pair':26s}{header}")
  for a in table:
    if not a.endswith("meta"):
      continue
    b = a[:-4] + "from_scratch"
    if b not in table:
      continue
    line = f"{a[:-5]:26s}"
    for i in range(len(BARS)):
      m, f = table[a][i], table[b][i]
      line += f"{f/m:>13.2f}" if m == m and f == f and m > 0 else f"{'--':>13s}"
    print(line)
