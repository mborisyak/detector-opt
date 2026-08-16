"""The ARM-PAIRED bar experiment: `meta` vs `from_scratch` on the SAME designs at TWO bars.

    python scripts/probe_armbar_report.py output/armbar

WHAT THIS EXISTS TO SETTLE. The documented 2.45x `from_scratch`/`meta` spend advantage was measured
at `loss_precision` 8.0e-3; every campaign at 1.0e-2 measures 1.3-1.7x. A DESIGN-paired proxy (one
arm, a small-gap design against a large-gap one) was run first and gave the OPPOSITE sign to the
model it was meant to test, so it settles nothing about the ARM ratio. This reads the arm-paired
measurement instead: both arms, the same seed -- hence the same `n_init` Sobol designs -- at both
bars.

THE QUANTITY is the arm ratio at each bar and how it CHANGES between them, taken as a WITHIN-DESIGN
DOUBLE DIFFERENCE so the design-to-design spread (0.59-3.50x on the campaign's own paired Sobol set)
cancels:

    d_i = log(spent_fs / spent_meta)|8.0e-3, design i   -   log(spent_fs / spent_meta)|1.0e-2, design i

COMPETING PREDICTIONS, stated before the run:
  * fixed-gap model (RETRACTED): the ratio EXPANDS as the bar tightens, by 1.4-2.2x.
  * gap-decay model (measured here: diff ~ w^-1.7..-2.2 against err ~ w^-0.5): the ratio COMPRESSES.
  * the documented pair (2.45x at 8.0e-3 against ~1.4x at 1.0e-2) implies EXPANSION by ~1.8x.
These differ in SIGN, so the double difference discriminates.

WHAT IS AND IS NOT PAIRED. The DESIGNS are identical across arms and bars (the first `n_init`
proposals are Sobol and depend only on the root seed). `meta`'s carried NETWORK is not: at design k it
depends on designs 1..k-1, which converged at different windows under a different bar. That is
intrinsic to the question -- it is what "the continual arm at another bar" means -- and only design 1
is fully state-matched. Reported per design so the reader can see it.

⚠️ THE 8.0e-3 CELLS RAN AT `training.budget=786432`, AND THAT MAKES THEIR VERDICT DRAW-CONDITIONAL.
`Trainer.__init__` draws ONE shuffled index of length `budget` and splits it at
`train_budget = budget - round(budget * val_fraction)`, so the budget fixes the VALIDATION set, and
the validation halves of two budgets are DISJOINT (verified: 5 shared events between 262144 and
786432; 47 between 786432 and 2097152 -- birthday collisions only). `diff` is |val - train|, so the
draw moves the quantity convergence is tested against. MEASURED across the first three designs, same
seed, same bar (1.0e-2), same width, identical design vectors -- budget 2097152 against 786432:

    design    diff (2097152 -> 786432)     window            spent            cost ratio
    1         0.00020 -> 0.00040             8192 ->   8192    10923 ->  10923      1.00x
    2         0.00120 -> 0.00780            36864 -> 446464    49150 -> 595250     12.11x
    3         0.00560 -> 0.00560            90112 -> 106496   120143 -> 141987      1.18x

THE EFFECT IS NOT UNIFORM, and that is the part worth carrying. Designs 1 and 3 barely move -- design
3 reports the SAME gap either way. Design 2 swings 12x in COST because its GAP swings 6.5x. The draw
bites exactly where the gap is marginal, i.e. exactly where the convergence decision is delicate, so
a budget change is near-harmless for comfortable designs and decisive for knife-edge ones. A probe
cannot know in advance which kind it has drawn, so budgets must MATCH between cells being compared
rather than be checked for "close enough" afterwards.

THE HAZARD HAS A BOUND ON THE DATA SIDE ONLY, and an earlier version of this note over-claimed it.
`budget_index` is built from (detector.size(), train_budget + val_budget, seed), so BOTH ARMS OF ONE
CELL share seed AND budget and get the IDENTICAL train/validation split. A comparison at ONE budget
is therefore FAIR -- the arms are matched on data. It does NOT follow that the arm RATIO is
draw-invariant, and it is not:

    design 3, from_scratch/meta:   1.10 at budget 2097152      2.89 at budget 786432

Design 2 (1.00 against 1.01) suggested invariance and was misleading -- it is a design where the two
arms coincide. The asymmetry has a mechanism, not noise. `from_scratch` is MEMORYLESS, so a draw
touches only the design being scored: its design-3 gap is 0.00560 under BOTH draws, identical.
`meta` is PATH-DEPENDENT -- network and replay pool carry designs 1..k-1 forward -- and the 786432
draw made design 2 cost 12x more, so meta reached design 3 holding 10x the history (600712 against
60073 calls of pool). More replay, smaller gap (0.00400 against 0.00540), cheaper design. That is the
advantage mechanism operating.

CONSEQUENCE FOR HOW ANY OF THIS IS QUOTED: a per-design arm ratio is a property of
(design, draw, trajectory), not of the design. Ratios are comparable only between cells sharing a
budget, and even then a `meta` number carries its own history. This makes the 2.45x worse, not
better: it compared `from_scratch` at 786432 against `meta` at 262144 -- disjoint validation sets
(5 shared events out of 196608 and 65536) AND a `meta` trajectory built on its own separate draw.

So "both arms cap at 8.0e-3 on design 2, and `diff + err` has a minimum near 0.0084" is, AS
MEASURED HERE, a statement about design 2 UNDER THE 786432 DRAW -- not about the task. Under the
campaign's own draw that design converges at 1.0e-2 with room to spare (window 36864, diff+err
0.00840), so its minimum may sit below 8.0e-3 and it might not cap at all. The MECHANISM (`err`
falling as w^-0.5 while `diff` rises over large windows, so their sum has a minimum) is not
draw-specific in kind; the NUMBER is, and so is whether 8.0e-3 sits above or below it for a given
design. Settling that needs the pair re-run at budget 2097152.

⚠️ THAT RE-RUN WAS SUBMITTED (jobs 1115/1116) AND THEN CANCELLED BEFORE IT STARTED, so the question
is CLOSED UNRESOLVED -- by decision, not by measurement. Nothing here licenses the unqualified claim
"8.0e-3 is infeasible for this task, measured". What IS established is narrower and still useful:
8.0e-3 was unreachable for BOTH arms on design 2 under the 786432 draw, and `diff + err` having a
minimum is a structural consequence of `err ~ w^-0.5` against a `diff` that rises over large windows.
Independent evidence pointing the same way, none of it at the campaign draw either: decision-log D22
found 8.0e-3 infeasible at `iteration_limit` 131072, and the original cost run's two crashes were
both `from_scratch` failing that bar. Treat "the task cannot reach 8.0e-3" as PLAUSIBLE AND UNTESTED
at the campaign budget until someone runs it.

CAPPED DESIGNS ARE A RESULT, NOT A FAILURE. `decision-log.md` D22 records 8.0e-3 as not feasible, and
the two designs dropped from the original 2.45x were `from_scratch` failing to clear it. `bo.py`
RAISES on a design that cannot reach precision, which ends that run, so a cell may hold fewer designs
than the others. Both are reported: the ratio over designs where BOTH arms converged, and the ratio
counting a capped design at its cap (`iteration_limit` events, the charge the original measurement
declined to pay).
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_meta_decompose import parse  # noqa: E402

ARMS = ("from_scratch", "meta")


def trajectory(run_dir):
  """The per-design rows a run wrote, finished (`results.json`) or not (`partial.json`)."""
  for name in ("results.json", "partial.json"):
    path = os.path.join(run_dir, name)
    if os.path.exists(path):
      with open(path) as f:
        payload = json.load(f)
      return payload.get("results", []), name == "results.json"
  return [], False


def margins(log_path):
  """`diff`/`err`/`window`/`rounds` per design, from the run log, plus the capped design if any."""
  if log_path is None or not os.path.exists(log_path):
    return [], None
  rows = parse(log_path)
  capped = None
  with open(log_path) as f:
    text = f.read()
  marker = "did not reach precision within iteration_limit"
  if marker in text:
    line = [ln for ln in text.splitlines() if marker in ln]
    capped = line[0].strip() if len(line) > 0 else marker
  return rows, capped


def cell(root, bar, seed, arm):
  run_dir = os.path.join(root, f"prec{bar}", str(seed), arm)
  rows, finished = trajectory(run_dir)
  log = f"logs/armbar-{bar}-{arm}.log"
  marg, capped = margins(log)
  return {"dir": run_dir, "rows": rows, "finished": finished, "margins": marg, "capped": capped}


def report(root, seed, bars, n_paired=5):
  cells = {(b, a): cell(root, b, seed, a) for b in bars for a in ARMS}

  print(f"ARM-PAIRED BAR EXPERIMENT  seed={seed}  root={root}\n")
  for (b, a), c in sorted(cells.items()):
    n = len(c["rows"])
    state = "finished" if c["finished"] else ("CAPPED/died" if c["capped"] is not None else "running/partial")
    print(f"  bar {b} {a:13s} designs={n:3d}  {state}")
    if c["capped"] is not None:
      print(f"      cap: {c['capped'][:150]}")
  print()

  print("PER-DESIGN, the shared Sobol prefix (design identity verified across every cell)")
  header = f"{'#':>2s}"
  for b in bars:
    header += f" | {('bar ' + b):>34s}"
  print(header)
  print(f"{'':>2s}" + (f" | {'fs spent':>10s} {'meta spent':>10s} {'ratio':>10s}") * len(bars))

  per_design = {b: [] for b in bars}
  for i in range(n_paired):
    line = f"{i+1:2d}"
    ok = True
    for b in bars:
      f_rows, m_rows = cells[(b, "from_scratch")]["rows"], cells[(b, "meta")]["rows"]
      if len(f_rows) <= i or len(m_rows) <= i:
        line += f" | {'--':>10s} {'--':>10s} {'--':>10s}"
        ok = False
        continue
      if f_rows[i].get("design") != m_rows[i].get("design"):
        line += f" | {'DESIGNS DIFFER':>32s}"
        ok = False
        continue
      fs, ms = f_rows[i]["spent"], m_rows[i]["spent"]
      per_design[b].append((i, fs, ms))
      line += f" | {fs:10d} {ms:10d} {fs/ms:10.2f}"
    print(line)
    del ok

  print()
  ratios = {}
  for b in bars:
    d = per_design[b]
    if len(d) == 0:
      continue
    fs = [x[1] for x in d]
    ms = [x[2] for x in d]
    ratios[b] = {
      "n": len(d),
      "median_of_ratios": st.median([a / c for _, a, c in d]),
      "ratio_of_medians": st.median(fs) / st.median(ms),
      "ratio_of_means": st.mean(fs) / st.mean(ms),
    }
    r = ratios[b]
    print(
      f"  bar {b}: n={r['n']}  median-of-ratios {r['median_of_ratios']:.2f}  "
      f"ratio-of-medians {r['ratio_of_medians']:.2f}  ratio-of-means {r['ratio_of_means']:.2f}"
    )

  if len(bars) == 2 and all(b in ratios for b in bars):
    lo, hi = bars  # lo = tighter bar
    common = {i for i, _, _ in per_design[lo]} & {i for i, _, _ in per_design[hi]}
    dd = []
    for i in sorted(common):
      a = next(x for x in per_design[lo] if x[0] == i)
      c = next(x for x in per_design[hi] if x[0] == i)
      dd.append(math.log((a[1] / a[2]) / (c[1] / c[2])))
    if len(dd) > 0:
      med = st.median(dd)
      sem = st.pstdev(dd) / math.sqrt(len(dd)) if len(dd) > 1 else float("nan")
      print(f"\n  WITHIN-DESIGN DOUBLE DIFFERENCE over {len(dd)} designs:")
      print(f"    median log-change of the arm ratio, bar {hi} -> {lo}: {med:+.3f}  "
            f"(x{math.exp(med):.2f}), sem {sem:.3f}")
      verdict = (
        "EXPANDS -- consistent with the documented 2.45x and with the retracted fixed-gap model"
        if med > 0 else "COMPRESSES -- consistent with the measured gap decay, NOT with 2.45x"
      )
      print(f"    => the arm ratio {verdict}")

  print("\nMARGINS at convergence (from the run logs)")
  for (b, a), c in sorted(cells.items()):
    m = [x for x in c["margins"] if x.get("diff") is not None][:n_paired]
    if len(m) == 0:
      continue
    print(
      f"  bar {b} {a:13s} diff med={st.median([x['diff'] for x in m]):.5f} "
      f"err med={st.median([x['err'] for x in m]):.5f} "
      f"window med={st.median([x['window'] for x in m]):8.0f} "
      f"rounds med={st.median([x['rounds'] for x in m]):5.1f}"
    )


if __name__ == "__main__":
  root = sys.argv[1] if len(sys.argv) > 1 else "output/armbar"
  seeds = sorted({os.path.basename(p) for p in glob.glob(os.path.join(root, "prec*", "*")) if os.path.isdir(p)})
  bars = sorted({os.path.basename(p)[4:] for p in glob.glob(os.path.join(root, "prec*"))})
  for s in seeds:
    report(root, s, bars)
