"""Where the meta advantage comes from, and why it shrank: the PRECISION HEADROOM model.

A design stops when its train/validation GAP plus its statistical ERROR fits under `loss_precision`.
`err` falls as c/sqrt(window) while `diff` is a bias floor that does not, so the window a design needs
is

    window ~= (c / (precision - diff))**2

and the cost ratio between two arms on the same design is the SQUARE of the ratio of their headrooms
`precision - diff`, times the square of the ratio of their noise constants `c`.

This script reads bo.py run logs, fits `c` per arm from the (window, err) pairs the log records, and
reports the headroom decomposition against the measured per-design cost. Read-only.

Usage: python scripts/probe_meta_headroom.py <label>=<log> [<label>=<log> ...]
"""

from __future__ import annotations

import math
import statistics as st
import sys

from probe_meta_decompose import parse


def report(label, path):
  d = [x for x in parse(path) if x.get("diff") is not None and x.get("spent") is not None]
  if len(d) == 0:
    print(f"{label}: nothing scored")
    return None
  prec = st.median([x["prec"] for x in d])
  diff = [x["diff"] for x in d]
  err = [x["err"] for x in d]
  win = [x["window"] for x in d]
  spent = [x["spent"] for x in d]
  # c = err * sqrt(window), the per-event loss noise the window has to beat down.
  c = [e * math.sqrt(w) for e, w in zip(err, win)]
  headroom = [prec - x for x in diff]
  # what the model predicts each design's window should have been
  pred = [(cc / h)**2 for cc, h in zip(c, headroom)]
  print(f"== {label}  n={len(d)} prec={prec:.4f}")
  print(f"   diff      med={st.median(diff):.5f}  as fraction of prec = {st.median(diff)/prec:.2f}")
  print(f"   headroom  med={st.median(headroom):.5f}  as fraction of prec = {st.median(headroom)/prec:.2f}")
  print(f"   err       med={st.median(err):.5f}")
  print(f"   c=err*sqrt(win)  med={st.median(c):.4f}  mean={st.mean(c):.4f}  sd={st.pstdev(c):.4f}")
  print(f"   window    med={st.median(win):8.0f}   model window med={st.median(pred):8.0f}")
  print(f"   spent     med={st.median(spent):8.0f}  mean={st.mean(spent):8.0f}")
  return dict(label=label, prec=prec, diff=st.median(diff), headroom=st.median(headroom), c=st.median(c),
              window=st.median(win), spent=st.median(spent), spent_mean=st.mean(spent), n=len(d))


if __name__ == "__main__":
  rows = {}
  for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    r = report(label, path)
    if r is not None:
      rows[label] = r
  print()
  print("PAIRED HEADROOM MODEL  (meta vs from_scratch, matched run)")
  print(f"{'pair':34s} {'measured':>9s} {'headroom^2':>11s} {'c^2':>7s} {'product':>8s}")
  for a in rows:
    if not a.endswith("meta"):
      continue
    b = a[:-4] + "from_scratch"
    if b not in rows:
      continue
    m, f = rows[a], rows[b]
    h = (m["headroom"] / f["headroom"])**2
    cc = (f["c"] / m["c"])**2
    print(f"{a[:-5]:34s} {f['spent']/m['spent']:9.2f} {h:11.2f} {cc:7.2f} {h*cc:8.2f}")
