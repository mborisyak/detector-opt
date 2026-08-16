"""Read-only decomposition of per-design training cost from bo.py run logs.

Parses the `[grow] window -> N`, `[converged]`/`[converged/bayes]` and `[iter k] ... spent=` lines a
`bo.py` run writes, and reports per design: how many GROWTH ROUNDS it took, the window it stopped at,
the detector calls it spent, and the margins (`diff`, `err`, and for the Bayesian rule `P(gap>LP)` and
`P(settled)`) that stopped it. Used to compare the `meta` arm against `from_scratch` on WHERE the cost
goes rather than on the total alone.

Usage: python scripts/probe_meta_decompose.py <log> [<log> ...]
"""

from __future__ import annotations

import re
import statistics as st
import sys

GROW = re.compile(r"\[grow\] window -> (\d+)")
SLOPE = re.compile(
  r"\[converged\] train=([\d.]+) val=([\d.]+) diff=([\d.]+) err=([\d.]+) "
  r"diff\+err=([\d.]+) prec=([\d.]+) \| window=(\d+)"
)
BAYES = re.compile(
  r"\[converged/bayes\] train=([\d.]+) val=([\d.]+) diff=([\d.]+) err=([\d.]+) "
  r"prec=([\d.]+) \| P\(gap>LP\)=([\d.]+) P\(settled\)=([\d.]+) \| window=(\d+)"
)
ITER = re.compile(r"\[iter (\d+)\] loss=([\d.]+)±([\d.]+) spent=(\d+) time=([\d.]+)s")
START = re.compile(r"\[iter (\d+)\] training\.\.\.")


def parse(path):
  """One record per design: rounds, window, spent, diff, err, rule and its margins."""
  designs, current = [], None
  for line in open(path):
    m = START.search(line)
    if m is not None:
      current = {"iter": int(m.group(1)), "rounds": 0, "window": None, "spent": None}
      continue
    if current is None:
      continue
    if GROW.search(line) is not None:
      current["rounds"] += 1
      current["window"] = int(GROW.search(line).group(1))
      continue
    m = SLOPE.search(line)
    if m is not None:
      current.update(
        rule="slope", train=float(m.group(1)), val=float(m.group(2)), diff=float(m.group(3)), err=float(m.group(4)),
        prec=float(m.group(6)), window=int(m.group(7))
      )
      continue
    m = BAYES.search(line)
    if m is not None:
      current.update(
        rule="bayes", train=float(m.group(1)), val=float(m.group(2)), diff=float(m.group(3)), err=float(m.group(4)),
        prec=float(m.group(5)), p_gap=float(m.group(6)), p_settled=float(m.group(7)), window=int(m.group(8))
      )
      continue
    m = ITER.search(line)
    if m is not None:
      current["spent"] = int(m.group(4))
      current["time"] = float(m.group(5))
      designs.append(current)
      current = None
  return designs


def summarise(path):
  d = parse(path)
  scored = [x for x in d if x.get("diff") is not None]
  if len(d) == 0:
    print(f"{path}: nothing parsed")
    return
  spent = [x["spent"] for x in d if x["spent"] is not None]
  rounds = [x["rounds"] for x in d]
  window = [x["window"] for x in d if x["window"] is not None]
  rules = {x.get("rule") for x in scored}
  print(f"== {path}")
  print(f"   designs={len(d)} rule={rules or '?'} prec={ {x.get('prec') for x in scored} }")
  print(f"   spent  mean={st.mean(spent):8.0f} med={st.median(spent):8.0f} min={min(spent):7d} max={max(spent):7d}")
  print(f"   rounds mean={st.mean(rounds):8.1f} med={st.median(rounds):8.1f} min={min(rounds):7d} max={max(rounds):7d}")
  print(f"   window mean={st.mean(window):8.0f} med={st.median(window):8.0f} min={min(window):7d} max={max(window):7d}")
  if len(scored) > 0:
    diff = [x["diff"] for x in scored]
    err = [x["err"] for x in scored]
    print(f"   diff   mean={st.mean(diff):.5f} med={st.median(diff):.5f} max={max(diff):.5f}")
    print(f"   err    mean={st.mean(err):.5f} med={st.median(err):.5f} max={max(err):.5f}")
    de = [a + b for a, b in zip(diff, err)]
    print(f"   diff+err med={st.median(de):.5f}  diff share med={st.median([a / c for a, c in zip(diff, de)]):.2f}")
    if "bayes" in rules:
      pg = [x["p_gap"] for x in scored if "p_gap" in x]
      ps = [x["p_settled"] for x in scored if "p_settled" in x]
      print(f"   P(gap>LP) med={st.median(pg):.3f} min={min(pg):.3f} max={max(pg):.3f}")
      print(f"   P(settled) med={st.median(ps):.3f} min={min(ps):.3f} max={max(ps):.3f}")
  print(f"   per-design spent: {spent}")
  print(f"   per-design rounds: {rounds}")


if __name__ == "__main__":
  for p in sys.argv[1:]:
    summarise(p)
