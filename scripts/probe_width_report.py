"""Does the `meta` advantage grow or shrink with NETWORK WIDTH? -- arm-paired, at one bar.

    python scripts/probe_width_report.py

WHAT IS COMPARED. The same arm-paired measurement at two widths, everything else held: seed
1244111331, `loss_precision` 1.0e-2, `budget` 2097152, the shipped `config/enzyme_extremes.yaml`
otherwise, and the same `n_init` Sobol designs (they depend only on the root seed, so they pair
across arms AND across widths).

  shipped   features [[24,16],[16,24]]   12648 parameters   output/enzyme_extremes/  (the campaign)
  1.5x      features [[36,24],[24,36]]   25112 parameters   output/armbar-wide/prec0.010-b2097152/

1.5x in WIDTH units is 1.99x in PARAMETERS -- stated because a width factor does not map to a
parameter factor by any rule worth trusting (the two blocks have different shapes, every linear
carries a bias, the learnable activation adds two gains per unit, and `n_models` multiplies all of
it). Both counts are measured, not derived.

WHY WIDTH IS THE RIGHT KNOB HERE. Convergence is `diff + err <= loss_precision`; `err` falls as
`w^-0.5` (measured: 3.7x over a 14x window growth against 3.8x predicted) and `meta` wins by carrying
a smaller `diff`, so it clears the bar at a smaller window. Capacity acts directly on `diff` -- a
wider network has more room to overfit a small window -- so it is the one knob that moves the term
the whole advantage runs through.

THE HEADLINE is the per-design CHANGE in the arm ratio between widths, a within-design double
difference so the design-to-design spread (1.00-3.50x at the shipped width) cancels:

    d_i = log(spent_fs / spent_meta)|1.5x, design i  -  log(spent_fs / spent_meta)|shipped, design i

FLAGGED SEPARATELY AND FIRST, because it is first-order: any design that CAPS at 1.0e-2 at the wider
width where it converged at the shipped width. `diff + err` has a minimum (measured ~0.0084 on design
2, only 16% under the 1.0e-2 bar), so extra capacity raising `diff` can push a design's minimum above
a bar it previously met. That would make the design unscoreable rather than merely dearer.
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_meta_decompose import parse  # noqa: E402

SEED = "1244111331"
# BOTH cells MUST share `training.budget`. `Trainer.__init__` draws ONE shuffled index of length
# `budget` and splits it at `train_budget = budget - round(budget * val_fraction)`, so the budget
# fixes both pools -- and the VALIDATION halves of two different budgets are DISJOINT (verified: 47
# shared events between budget 786432 and 2097152, birthday collisions only, while the TRAIN halves
# share a prefix). `diff` is |val - train|, the term this whole study runs through, so a cell at
# another budget measures a different validation set and is NOT a comparator. Measured consequence
# at the shipped width, same seed, same bar, design 2: 49150 calls at budget 2097152 against 595250
# at 786432 -- a 12x swing from the draw alone.
# The shipped-width arm is therefore the CAMPAIGN run (budget 2097152, read only, never written) and
# the wide cells are submitted at that same budget to match its draw.
WIDTHS = (("shipped", "output/enzyme_extremes", None, 12648),
          ("wide1.5x", "output/armbar-wide/prec0.010-b2097152", "logs/armbarwide2-{arm}.log", 25112),
          )
SHIPPED_LOGS = {
  "from_scratch": "output/slurm-log-archive/rule_bo_output_enzyme_extremes_1244111331_from_scratch_1015.log",
  "meta": "output/slurm-log-archive/rule_bo_output_enzyme_extremes_1244111331_meta_1017.log",
}
ARMS = ("from_scratch", "meta")
CAP_MARKER = "did not reach precision within iteration_limit"

# PRE-DECLARED, before any wide-width number existed, and not to be moved after seeing results.
# A window is n0 + k * n_increment = 8192 + k * 4096, so a single cell is quantised to about +-7%;
# a ratio of ratios carries four such cells, ~+-0.14 in log. With the effective n of 3 stated in
# advance that is a sem near 0.08, so the smallest change in the arm ratio this design can support
# is about 1.2x, i.e. |log| > 0.18. Below that the honest report is a NULL, never a direction.
QUANTISATION_LOG = 0.07
RESOLUTION_LOG = 0.18


def rows_for(root, arm, log_template):
  log = SHIPPED_LOGS[arm] if log_template is None else log_template.format(arm=arm)
  run_dir = os.path.join(root, SEED, arm)
  results, finished = [], False
  for name in ("results.json", "partial.json"):
    path = os.path.join(run_dir, name)
    if os.path.exists(path):
      with open(path) as f:
        results = json.load(f).get("results", [])
      finished = name == "results.json"
      break
  margins = parse(log) if os.path.exists(log) else []
  capped = None
  if os.path.exists(log):
    with open(log) as f:
      text = f.read()
    if CAP_MARKER in text:
      capped = next((ln.strip() for ln in text.splitlines() if CAP_MARKER in ln), CAP_MARKER)
  return results, margins, capped, finished


def main(n_paired=5):
  data = {}
  print("WIDTH STUDY -- meta vs from_scratch at loss_precision 1.0e-2, seed 1244111331, budget 2097152\n")
  caps = []
  for label, root, log_t, n_par in WIDTHS:
    for arm in ARMS:
      res, marg, capped, finished = rows_for(root, arm, log_t)
      data[(label, arm)] = (res, marg)
      state = "finished" if finished else ("CAPPED/died" if capped is not None else "running/partial")
      print(f"  {label:9s} {arm:13s} {n_par:6d} params  designs={len(res):3d}  {state}")
      if capped is not None:
        caps.append((label, arm, capped))
  if len(caps) > 0:
    print("\n  *** CAP AT 1.0e-2 -- FIRST-ORDER RESULT ***")
    for label, arm, line in caps:
      print(f"    {label} {arm}: {line[:160]}")
  print()

  print("PER DESIGN")
  head = f"{'#':>2s}"
  for label, _, _, _ in WIDTHS:
    head += f" | {label + ': fs spent':>16s} {'meta spent':>10s} {'ratio':>6s}"
  print(head)
  ratios = {label: [] for label, _, _, _ in WIDTHS}
  for i in range(n_paired):
    line = f"{i+1:2d}"
    for label, _, _, _ in WIDTHS:
      f_res = data[(label, "from_scratch")][0]
      m_res = data[(label, "meta")][0]
      if len(f_res) <= i or len(m_res) <= i:
        line += f" | {'--':>16s} {'--':>10s} {'--':>6s}"
        continue
      if f_res[i].get("design") != m_res[i].get("design"):
        line += f" | {'DESIGNS DIFFER':>34s}"
        continue
      fs, ms = f_res[i]["spent"], m_res[i]["spent"]
      ratios[label].append((i, fs, ms))
      line += f" | {fs:16d} {ms:10d} {ratios[label][-1][1]/ms:6.2f}"
    print(line)

  print()
  for label in ratios:
    d = ratios[label]
    if len(d) == 0:
      continue
    per = [a / c for _, a, c in d]
    print(
      f"  {label:9s} n={len(d)}  median-of-ratios {st.median(per):.2f}  "
      f"ratio-of-medians {st.median([a for _, a, _ in d])/st.median([c for _, _, c in d]):.2f}  "
      f"ratio-of-means {st.mean([a for _, a, _ in d])/st.mean([c for _, _, c in d]):.2f}"
    )

  a, b = [w[0] for w in WIDTHS]
  common = {i for i, _, _ in ratios[a]} & {i for i, _, _ in ratios[b]}
  if len(common) > 0:
    dd = []
    for i in sorted(common):
      x = next(t for t in ratios[b] if t[0] == i)
      y = next(t for t in ratios[a] if t[0] == i)
      dd.append(math.log((x[1] / x[2]) / (y[1] / y[2])))
    med = st.median(dd)
    sem = st.pstdev(dd) / math.sqrt(len(dd)) if len(dd) > 1 else float("nan")
    # An UNINFORMATIVE design is one where BOTH widths report an arm ratio of 1.00 to within the
    # window quantisation -- the arms coincide there, so it contributes an exact zero and DILUTES the
    # median toward "no change" without carrying evidence either way. Designs 1 and 2 were stated to
    # be of this kind BEFORE the run (design 1 converges at n0 with zero growth rounds, so its spend
    # is fixed at 10923 at any width), which is why the effective n was pre-declared as 3, not 5.
    informative = [
      i for i in sorted(common)
      if abs(math.log(next(t for t in ratios[a] if t[0] == i)[1] / next(t
                                                                        for t in ratios[a] if t[0] == i)[2])) > QUANTISATION_LOG
    ]
    dd_informative = [dd[k] for k, i in enumerate(sorted(common)) if i in informative]
    print(f"\n  WITHIN-DESIGN DOUBLE DIFFERENCE over {len(dd)} designs, {a} -> {b}:")
    print(f"    median log-change of the arm ratio {med:+.3f} (x{math.exp(med):.2f}), sem {sem:.3f}")
    print(
      f"    informative designs (arms differ at the shipped width): "
      f"{[i + 1 for i in informative] if len(informative) > 0 else 'NONE'}"
    )
    if len(dd_informative) == 0:
      print("    => NOT RESOLVED: no informative design yet. The zeros above come from designs where")
      print("       the two arms coincide by construction; they are not evidence of 'no effect'.")
    else:
      med_i = st.median(dd_informative)
      signs = {d > 0 for d in dd_informative}
      consistent = len(signs) == 1
      print(
        f"    over the {len(dd_informative)} informative design(s): median {med_i:+.3f} "
        f"(x{math.exp(med_i):.2f}), signs {'consistent' if consistent else 'INCONSISTENT'}"
      )
      if abs(med_i) <= RESOLUTION_LOG or not consistent:
        print(
          f"    => NOT RESOLVED at this power: |{med_i:+.3f}| does not clear the pre-declared "
          f"threshold {RESOLUTION_LOG:.2f}"
        )
        print("       (or the per-design signs disagree). Report a null, NOT a direction.")
      else:
        print(f"    => RESOLVED: the meta advantage {'GROWS' if med_i > 0 else 'SHRINKS'} with width")

  print("\nMARGINS AT STOPPING (the term the advantage runs through)")
  for label, _, _, _ in WIDTHS:
    for arm in ARMS:
      m = [x for x in data[(label, arm)][1] if x.get("diff") is not None][:n_paired]
      if len(m) == 0:
        continue
      print(
        f"  {label:9s} {arm:13s} diff med={st.median([x['diff'] for x in m]):.5f} "
        f"err med={st.median([x['err'] for x in m]):.5f} "
        f"window med={st.median([x['window'] for x in m]):8.0f} "
        f"rounds med={st.median([x['rounds'] for x in m]):5.1f}"
      )


if __name__ == "__main__":
  main()
