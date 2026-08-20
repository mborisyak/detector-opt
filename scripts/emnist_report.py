#!/usr/bin/env python3
"""Campaign report for the EMNIST occlusion arm comparison: summary table, per-seed convergence
grid, and the paired replay-weight probe.

    python scripts/emnist_report.py --campaign output/campaign-emnist \
        --fill output/campaign-emnist-cloud --output-dir output/campaign-emnist/plots

    python scripts/emnist_report.py --campaign output/campaign-emnist \
        --fill output/campaign-emnist-cloud --replay output/campaign-emnist-rw025 \
        --output-dir output/campaign-emnist-rw025/plots --only-replay

A CELL is one ``<root>/<seed>/<arm>/results.json``. A cell is ADMITTED only when its ``completed``
field is true and it holds at least one scored row: a run killed mid-trajectory has a truthful
best-so-far for the designs it reached, but its endpoint is not an endpoint and must never enter a
median beside finished ones. Rejected cells are printed with their reason and appear in the table
marked, never silently dropped.

``--fill`` supplies FALLBACK roots for seeds ``--campaign`` did not finish. It does not pool: a
replicate of a cell the primary already has is ignored, because two runs of the same seed are one
seed and averaging them would understate the spread. Filling is resolved PER SEED, not per cell --
a root that supplies every arm of a seed is taken whole, and only when none does is the seed
assembled cell by cell. The arm order within a seed is the comparison, so a source change inside one
seed row would put a machine difference exactly where an arm difference is supposed to be. The table
records which root every row came from.

Reported loss is SELF-EVALUATED -- the convergence procedure stopped on the very train/validation
losses it hands to BO. Where ``verification.json`` (``scripts/verify_trajectory.py``) sits beside a
``results.json`` its held-out re-score is the trustworthy number, and the table carries it in its own
column; the two are never merged into one figure.

``--replicate`` names a root expected to hold RE-RUNS of cells the primary already has, and answers
whether the two roots are the same configuration at all. The evidence is the leading designs: those
are proposed before the surrogate has any data, so under one configuration and one seed the two roots
must propose them BIT-IDENTICALLY, spend the same per design, and score them within run-to-run noise.
Where that holds the roots are interchangeable; where it fails they are two different experiments and
must not share an axis.

``--replay`` pairs a meta-only variant campaign against the primary's ``meta`` AT THE SAME SEED. The
first ``n_init`` designs are drawn before the surrogate has any data, so where the two runs propose
them identically they form a paired sample and a difference there is the setting's, not the design's.
That prefix is detected by comparing ``x_scaled``, not assumed. The FIRST design is the null
calibration point: no replay buffer exists yet, so ``replay_weight`` cannot act on it and whatever
difference appears there is the run-to-run noise floor.

Colours follow the arm (or the seed, in the replay figure), never rank, and carry a distinct marker
as secondary encoding; the palette is the project's validated categorical order in
``detopt.utils.viz.bo``.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("AGG")

import numpy as np
from matplotlib.figure import Figure

from detopt.utils import io
from detopt.utils.viz.bo import _STRATEGY_STYLE

ARM_ORDER = ("from_scratch", "continue", "closest", "meta")
UNIFORM_CROSS_ENTROPY = float(np.log(10.0))


def load_cell(root, seed, arm):
  """The admitted cell at ``<root>/<seed>/<arm>``, or ``(None, reason)`` when it is not admissible."""
  directory = os.path.join(root, seed, arm)
  path = os.path.join(directory, "results.json")
  if not os.path.exists(path):
    return None, "no results.json"
  with open(path) as handle:
    payload = json.load(handle)
  rows = io.complete_results(payload.get("results", []))
  if len(rows) == 0:
    return None, "no scored rows"
  if payload.get("completed") is not True:
    return None, f"completed=False, stopped after {len(rows)} design(s)"
  verification = None
  verification_path = os.path.join(directory, "verification.json")
  if os.path.exists(verification_path):
    with open(verification_path) as handle:
      verification = json.load(handle)
  return {"rows": rows, "payload": payload, "verification": verification, "directory": directory}, None


def resolve(roots, seed, arm):
  """First admitted cell for ``(seed, arm)`` over ``roots`` in order, plus every rejection seen."""
  rejected = []
  for root in roots:
    cell, reason = load_cell(root, seed, arm)
    if cell is not None:
      return cell, root, rejected
    if reason != "no results.json":
      rejected.append((root, reason))
  return None, None, rejected


def resolve_seed(roots, seed, arms):
  """Roots reordered so a root that supplies EVERY arm of this seed is consulted first.

  Arm order within a seed is the comparison; a source change inside one seed row puts a
  machine difference where an arm difference is supposed to be. Two replicates of the same
  finished cell here disagree at the endpoint by as much as the arms do, so a whole-seed
  source is preferred and per-cell filling is only the fallback."""
  complete = [r for r in roots if all(load_cell(r, seed, a)[0] is not None for a in arms)]
  return complete + [r for r in roots if r not in complete]


def seed_directories(roots, arms):
  """Sub-directories of ``roots`` that actually look like a seed: at least one arm inside."""
  found = set()
  for root in roots:
    for entry in os.listdir(root):
      if entry.startswith(".") or not os.path.isdir(os.path.join(root, entry)):
        continue
      if any(os.path.isdir(os.path.join(root, entry, a)) for a in arms):
        found.add(entry)
  return sorted(found)


def curve(rows):
  """Cumulative detector calls and best-so-far reported loss over a trajectory."""
  calls = np.cumsum([r["spent"] for r in rows], dtype=np.float64)
  best = np.minimum.accumulate(np.asarray([r["loss"] for r in rows], dtype=np.float64))
  return calls, best


def verified_endpoint(verification):
  """Best held-out test loss and its SEM over the verified incumbents, or ``(None, None)``."""
  points = (verification or {}).get("points") or []
  if len(points) == 0:
    return None, None
  index = int(np.argmin([p["test_loss"] for p in points]))
  return float(points[index]["test_loss"]), float(points[index]["test_sem"])


def build_table(roots, seeds, arms):
  """One record per (seed, arm), resolved over ``roots``, with its rejections attached."""
  table = []
  for seed in seeds:
    ordered = resolve_seed(roots, seed, arms)
    for arm in arms:
      cell, source, rejected = resolve(ordered, seed, arm)
      record = {"seed": seed, "arm": arm, "source": source, "rejected": rejected}
      if cell is None:
        record.update(
          n_designs=0, final_loss=float("nan"), total_calls=0, calls_per_design=float("nan"), verified_loss=None,
          verified_sem=None, status="ABSENT" if len(rejected) == 0 else "EXCLUDED"
        )
      else:
        rows = cell["rows"]
        total = int(sum(r["spent"] for r in rows))
        test_loss, test_sem = verified_endpoint(cell["verification"])
        record.update(
          n_designs=len(rows), final_loss=float(min(r["loss"] for r in rows)), total_calls=total,
          calls_per_design=total / len(rows), verified_loss=test_loss, verified_sem=test_sem, status="ok", rows=rows
        )
      table.append(record)
  return table


def write_table(table, output_dir):
  """The summary table as TSV and as a fixed-width text block; returns both paths."""
  columns = (
    "seed", "arm", "source", "status", "n_designs", "final_loss", "verified_loss", "verified_sem", "total_calls",
    "calls_per_design"
  )
  tsv_path = os.path.join(output_dir, "summary.tsv")
  with open(tsv_path, "w") as handle:
    handle.write("\t".join(columns) + "\n")
    for record in table:
      handle.write("\t".join("" if record.get(c) is None else str(record.get(c, "")) for c in columns) + "\n")

  lines = [
    f"{'seed':>11}{'arm':>14}{'source':>26}{'status':>10}{'designs':>9}"
    f"{'reported':>10}{'verified':>10}{'calls':>9}{'calls/design':>14}"
  ]
  lines.append("-" * len(lines[0]))
  for record in table:
    source = "-" if record["source"] is None else os.path.basename(os.path.normpath(record["source"]))
    reported = "-" if not np.isfinite(record["final_loss"]) else f"{record['final_loss']:.4f}"
    verified = "-" if record["verified_loss"] is None else f"{record['verified_loss']:.4f}"
    per = "-" if not np.isfinite(record["calls_per_design"]) else f"{record['calls_per_design']:.0f}"
    lines.append(
      f"{record['seed']:>11}{record['arm']:>14}{source:>26}{record['status']:>10}"
      f"{record['n_designs']:>9}{reported:>10}{verified:>10}{record['total_calls']:>9}{per:>14}"
    )
  excluded = [(r["seed"], r["arm"], r["rejected"]) for r in table if len(r["rejected"]) > 0]
  if len(excluded) > 0:
    lines.append("")
    lines.append("EXCLUDED CELLS")
    for seed, arm, reasons in excluded:
      for root, reason in reasons:
        lines.append(f"  {seed}/{arm} at {root}: {reason}")
  text_path = os.path.join(output_dir, "summary.txt")
  with open(text_path, "w") as handle:
    handle.write("\n".join(lines) + "\n")
  print("\n".join(lines))
  return tsv_path, text_path


def write_inventory(roots, seeds, arms, output_dir):
  """Every ``<root>/<seed>/<arm>`` the campaign could have, admitted or not, as an audit trail."""
  lines = [f"{'root':>28}{'seed':>12}{'arm':>14}{'admitted':>10}{'designs':>9}{'reported':>10}{'note':>44}"]
  lines.append("-" * len(lines[0]))
  records = []
  for root in roots:
    for seed in seeds:
      for arm in arms:
        cell, reason = load_cell(root, seed, arm)
        if cell is None and reason == "no results.json":
          exists = os.path.isdir(os.path.join(root, seed, arm))
          if exists is False:
            continue
          reason = "directory present, empty -- job produced nothing"
        name = os.path.basename(os.path.normpath(root))
        if cell is None:
          lines.append(f"{name:>28}{seed:>12}{arm:>14}{'no':>10}{0:>9}{'-':>10}{reason:>44}")
          records.append({"root": root, "seed": seed, "arm": arm, "admitted": False, "note": reason})
        else:
          rows = cell["rows"]
          best = min(r["loss"] for r in rows)
          lines.append(f"{name:>28}{seed:>12}{arm:>14}{'yes':>10}{len(rows):>9}{best:>10.4f}{'':>44}")
          records.append({
            "root": root,
            "seed": seed,
            "arm": arm,
            "admitted": True,
            "n_designs": len(rows),
            "final_loss": float(best)
          })
  path = os.path.join(output_dir, "inventory.txt")
  with open(path, "w") as handle:
    handle.write("\n".join(lines) + "\n")
  print("\n".join(lines))
  return path, records


def load_rows(root, seed, arm):
  """The scored rows of a cell REGARDLESS of completion, for questions a truncated run can answer."""
  path = os.path.join(root, seed, arm, "results.json")
  if not os.path.exists(path):
    return None, None
  with open(path) as handle:
    payload = json.load(handle)
  rows = io.complete_results(payload.get("results", []))
  return (rows, payload) if len(rows) > 0 else (None, None)


def plot_replicate(primary, replicate, seeds, arms, output_dir):
  """Same (seed, arm) run twice on different hardware: do the two roots agree where they overlap?

  Cells the primary did not FINISH are admitted here and nowhere else. Completion is irrelevant to
  the question this figure asks -- whether the two roots ran the same experiment -- and the leading
  designs a truncated run did reach are exactly the evidence, so excluding it would throw away a
  comparison for a reason that does not apply.
  """
  figure = Figure(figsize=(13, 5.4))
  left, right = figure.subplots(1, 2)
  dumped = {}
  shared_deltas = []
  cell_index = 0

  for seed in seeds:
    for arm in arms:
      a_rows, a_payload = load_rows(primary, seed, arm)
      b_rows, b_payload = load_rows(replicate, seed, arm)
      if a_rows is None or b_rows is None:
        continue
      colour, marker = _STRATEGY_STYLE[cell_index % len(_STRATEGY_STYLE)]
      cell_index += 1
      a_calls, a_best = curve(a_rows)
      b_calls, b_best = curve(b_rows)
      truncated = "" if a_payload.get("completed") is True else " (primary truncated)"
      left.step(
        a_calls, a_best, where="post", lw=2.0, ls="-", color=colour, marker=marker, ms=5, label=f"{seed} {arm}{truncated}"
      )
      left.step(b_calls, b_best, where="post", lw=2.0, ls="--", color=colour, marker=marker, ms=5, mfc="none")
      k = paired_prefix(a_rows, b_rows)
      delta = [b_rows[i]["loss"] - a_rows[i]["loss"] for i in range(k)]
      spent_delta = [b_rows[i]["spent"] - a_rows[i]["spent"] for i in range(k)]
      right.plot(
        np.arange(1, k + 1), delta, ls="none", marker=marker, ms=8, color=colour, alpha=0.85, label=f"{seed} {arm} (k={k})"
      )
      shared_deltas.extend(delta)
      dumped[f"{seed}/{arm}"] = {
        "identical_leading_designs": int(k),
        "delta_reported_loss": delta,
        "delta_spent": spent_delta,
        "primary_completed": a_payload.get("completed") is True,
        "replicate_completed": b_payload.get("completed") is True,
        "n_designs": [len(a_rows), len(b_rows)],
        "final_reported": [float(a_best[-1]), float(b_best[-1])]
      }

  left.set_yscale("log")
  left.grid(True, alpha=0.25, lw=0.6)
  left.set_xlabel("cumulative detector calls")
  left.set_ylabel("best-so-far reported loss")
  left.set_title(
    f"solid = {os.path.basename(os.path.normpath(primary))}, "
    f"dashed = {os.path.basename(os.path.normpath(replicate))}"
  )
  if len(left.get_lines()) > 0:
    left.legend(loc="upper right", fontsize=7)
  right.axhline(0.0, color="#52514e", lw=1.2)
  right.grid(True, alpha=0.25, lw=0.6)
  right.set_xlabel("design index over the identical leading designs")
  right.set_ylabel("reported loss: replicate minus primary")
  headline = "" if len(shared_deltas) == 0 else \
      f"\nn={len(shared_deltas)} shared designs, max|delta| = {float(np.max(np.abs(shared_deltas))):.4f}"
  right.set_title(f"agreement on designs the two roots proposed identically{headline}", fontsize=10)
  if len(right.get_lines()) > 0:
    right.legend(loc="lower left", fontsize=7)

  figure.suptitle("Configuration comparability: identical seeds re-run on a second machine")
  figure.tight_layout()
  png_path = os.path.join(output_dir, "replicate-agreement.png")
  figure.savefig(png_path, dpi=130)
  json_path = os.path.join(output_dir, "replicate-agreement.json")
  with open(json_path, "w") as handle:
    json.dump({
      "primary": primary,
      "replicate": replicate,
      "cells": dumped,
      "max_abs_delta_shared": float(np.max(np.abs(shared_deltas))) if len(shared_deltas) > 0 else None,
      "n_shared_designs": len(shared_deltas)
    }, handle, indent=2, default=float)
  print(f"  [plot] replicate agreement -> {png_path}")
  return png_path, json_path


def rank_summary(table, seeds, arms):
  """Within-seed ranks of the final reported loss, averaged over the seeds that have every arm."""
  usable = [s for s in seeds if all(any(r["seed"] == s and r["arm"] == a and r["status"] == "ok" for r in table) for a in arms)]
  if len(usable) < 2:
    return usable, {}, {}
  by = {(r["seed"], r["arm"]): r["final_loss"] for r in table if r["status"] == "ok"}
  ranks = {a: [] for a in arms}
  for seed in usable:
    losses = np.asarray([by[(seed, a)] for a in arms], dtype=np.float64)
    order = np.argsort(np.argsort(losses)) + 1
    for arm, rank in zip(arms, order):
      ranks[arm].append(int(rank))
  mean = {a: float(np.mean(v)) for a, v in ranks.items()}
  sem = {a: float(np.std(v, ddof=1) / np.sqrt(len(v))) for a, v in ranks.items()}
  return usable, mean, sem


def plot_per_seed(table, seeds, arms, output_dir, title):
  """Best-so-far reported loss vs cumulative detector calls, four arms overlaid, one panel per seed."""
  columns = 3
  rows_count = int(np.ceil(len(seeds) / columns))
  figure = Figure(figsize=(5.2 * columns, 4.1 * rows_count))
  grid = figure.subplots(rows_count, columns, squeeze=False, sharey=True)
  axes = grid.ravel()
  dumped = {}

  for panel, seed in zip(axes, seeds):
    sources = set()
    for index, arm in enumerate(arms):
      record = next((r for r in table if r["seed"] == seed and r["arm"] == arm), None)
      if record is None or record["status"] != "ok":
        continue
      colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
      calls, best = curve(record["rows"])
      panel.step(calls, best, where="post", lw=2.0, marker=marker, ms=5, color=colour, label=arm)
      sources.add(os.path.basename(os.path.normpath(record["source"])))
      dumped.setdefault(seed, {})[arm] = {"calls": calls.tolist(), "best_so_far": best.tolist(), "source": record["source"]}
    panel.axhline(UNIFORM_CROSS_ENTROPY, color="#8a8884", ls=":", lw=1.2)
    panel.annotate(
      "ln(10): whole image = no information", xy=(0.02, UNIFORM_CROSS_ENTROPY), xycoords=("axes fraction", "data"), va="bottom",
      fontsize=7.5, color="#52514e"
    )
    panel.set_yscale("log")
    panel.grid(True, alpha=0.25, lw=0.6)
    panel.set_title(f"seed {seed}  [{', '.join(sorted(sources)) if len(sources) > 0 else 'no admitted cell'}]", fontsize=10)
    panel.set_xlabel("cumulative detector calls")
    if panel in list(grid[:, 0]):
      panel.set_ylabel("best-so-far reported loss")
    if len(panel.get_lines()) > 0:
      panel.legend(loc="upper right", fontsize=8)
  for panel in axes[len(seeds):]:
    panel.set_visible(False)

  figure.suptitle(title)
  figure.tight_layout()
  png_path = os.path.join(output_dir, "per-seed-convergence.png")
  figure.savefig(png_path, dpi=130)
  json_path = os.path.join(output_dir, "per-seed-convergence.json")
  with open(json_path, "w") as handle:
    json.dump({"seeds": dumped, "reference_line": UNIFORM_CROSS_ENTROPY}, handle, indent=2, default=float)
  print(f"  [plot] per-seed convergence -> {png_path}")
  return png_path, json_path


def paired_prefix(left, right):
  """Number of leading designs the two trajectories proposed identically."""
  k, limit = 0, min(len(left), len(right))
  while k < limit and np.allclose(left[k]["x_scaled"], right[k]["x_scaled"], atol=1e-6):
    k += 1
  return k


def plot_replay(baseline_roots, replay_root, seeds, output_dir):
  """rw=0.25 meta against rw=1.0 meta at the same seed: convergence, and the paired per-design delta.

  EVERY admitted rw=1.0 replicate of a seed is drawn, not a chosen one. Where a seed was run twice
  under the SAME setting the two endpoints bracket what a single comparison can claim, and hiding one
  of them would present a difference between replicates as a difference between settings.
  """
  figure = Figure(figsize=(13, 5.4))
  left, right = figure.subplots(1, 2)
  dumped = {}
  deltas_all = []
  per_seed_mean = {}

  for index, seed in enumerate(seeds):
    variant, _ = load_cell(replay_root, seed, "meta")
    if variant is None:
      continue
    colour, marker = _STRATEGY_STYLE[index % len(_STRATEGY_STYLE)]
    variant_calls, variant_best = curve(variant["rows"])
    left.step(
      variant_calls, variant_best, where="post", lw=2.2, ls="--", color=colour, marker=marker, ms=6, mfc="none", mew=1.6,
      label=f"{seed} rw=0.25"
    )
    entry = {
      "replay_weight_0.25": {
        "calls": variant_calls.tolist(),
        "best_so_far": variant_best.tolist(),
        "final": float(variant_best[-1]),
        "n_designs": len(variant["rows"])
      },
      "baselines": {}
    }
    means = []
    for style, root in zip(("-", (0, (1, 1))), baseline_roots):
      base, _ = load_cell(root, seed, "meta")
      if base is None:
        continue
      name = os.path.basename(os.path.normpath(root))
      base_calls, base_best = curve(base["rows"])
      left.step(
        base_calls, base_best, where="post", lw=1.8, ls=style, color=colour, marker=marker, ms=5, alpha=0.85,
        label=f"{seed} rw=1.0 [{name}]"
      )
      k = paired_prefix(base["rows"], variant["rows"])
      delta = np.asarray([variant["rows"][i]["loss"] - base["rows"][i]["loss"] for i in range(k)], dtype=np.float64)
      if k > 1:
        right.plot(
          np.arange(2, k + 1), delta[1:], ls=style, lw=1.8, marker=marker, ms=7, color=colour, label=f"{seed} vs {name}"
        )
        right.plot([1], delta[:1], ls="none", marker=marker, ms=10, mfc="none", mec=colour, mew=1.8)
        deltas_all.extend(delta[1:].tolist())
        means.append(float(np.mean(delta[1:])))
      entry["baselines"][name] = {
        "calls": base_calls.tolist(),
        "best_so_far": base_best.tolist(),
        "final": float(base_best[-1]),
        "n_designs": len(base["rows"]),
        "paired_designs": int(k),
        "delta_reported_loss": delta.tolist(),
        "delta_design_1_replay_inert": float(delta[0]) if k > 0 else None,
        "delta_mean_designs_2plus": float(np.mean(delta[1:])) if k > 1 else None
      }
    if len(means) > 0:
      per_seed_mean[seed] = float(np.mean(means))
    dumped[seed] = entry

  left.set_yscale("log")
  left.grid(True, alpha=0.25, lw=0.6)
  left.set_xlabel("cumulative detector calls")
  left.set_ylabel("best-so-far reported loss")
  left.set_title("meta: replay_weight 1.0 (filled) vs 0.25 (open)", fontsize=10)
  if len(left.get_lines()) > 0:
    left.legend(loc="lower left", fontsize=7)

  right.axhline(0.0, color="#52514e", lw=1.2)
  right.grid(True, alpha=0.25, lw=0.6)
  right.set_xlabel("design index over the identical leading designs")
  right.set_ylabel("reported loss: rw=0.25 minus rw=1.0")
  right.set_title("paired on identical designs\ndesign 1 is open: no replay buffer yet, so the setting cannot act", fontsize=10)
  seed_means = np.asarray(sorted(per_seed_mean.values()), dtype=np.float64)
  if seed_means.size > 1:
    grand = float(seed_means.mean())
    sem = float(seed_means.std(ddof=1) / np.sqrt(seed_means.size))
    right.axhline(grand, color="#52514e", ls="--", lw=1.2)
    right.axhspan(grand - sem, grand + sem, color="#52514e", alpha=0.10)
    right.annotate(
      f"mean of the {seed_means.size} per-seed means = {grand:+.4f} +/- {sem:.4f} (SEM over seeds)", xy=(0.02, grand),
      xycoords=("axes fraction", "data"), va="bottom", fontsize=8, color="#52514e"
    )
  if len(right.get_lines()) > 0:
    right.legend(loc="lower right", fontsize=7)

  figure.suptitle("EMNIST occlusion, meta arm: replay-weight probe, paired by seed")
  figure.tight_layout()
  png_path = os.path.join(output_dir, "replay-weight-paired.png")
  figure.savefig(png_path, dpi=130)
  json_path = os.path.join(output_dir, "replay-weight-paired.json")
  with open(json_path, "w") as handle:
    json.dump({
      "seeds": dumped,
      "per_seed_mean_delta_designs_2plus": per_seed_mean,
      "grand_mean_over_seeds": float(seed_means.mean()) if seed_means.size > 0 else None,
      "sem_over_seeds": float(seed_means.std(ddof=1) / np.sqrt(seed_means.size)) if seed_means.size > 1 else None,
      "delta_n_paired_designs": len(deltas_all)
    }, handle, indent=2, default=float)
  print(f"  [plot] replay-weight probe -> {png_path}")
  return png_path, json_path


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--campaign", required=True)
  parser.add_argument("--fill", nargs="*", default=[], help="fallback roots, used only where --campaign has no cell")
  parser.add_argument("--replay", default=None, help="meta-only variant campaign to pair against --campaign's meta")
  parser.add_argument("--replicate", default=None, help="root holding re-runs of --campaign's cells, for comparability")
  parser.add_argument("--seeds", nargs="*", default=None)
  parser.add_argument("--arms", nargs="*", default=list(ARM_ORDER))
  parser.add_argument("--output-dir", required=True)
  parser.add_argument("--only-replay", action="store_true")
  parser.add_argument("--title", default="EMNIST occlusion: best-so-far by arm")
  arguments = parser.parse_args()

  roots = [arguments.campaign] + list(arguments.fill)
  os.makedirs(arguments.output_dir, exist_ok=True)

  if arguments.seeds is None:
    seeds = seed_directories(roots, arguments.arms)
  else:
    seeds = list(arguments.seeds)

  written = []
  if arguments.only_replay is False:
    inventory_path, _ = write_inventory(roots, seeds, arguments.arms, arguments.output_dir)
    written.append(inventory_path)
    print()
    table = build_table(roots, seeds, arguments.arms)
    written.extend(write_table(table, arguments.output_dir))
    usable, mean, sem = rank_summary(table, seeds, arguments.arms)
    if len(mean) > 0:
      chance = (len(arguments.arms) + 1) / 2.0
      print(
        f"\nMEAN WITHIN-SEED RANK of final reported loss over {len(usable)} complete seeds "
        f"(1 = best; chance = {chance:.1f})"
      )
      for arm in arguments.arms:
        print(f"  {arm:>14}  {mean[arm]:.2f} +/- {sem[arm]:.2f}")
    written.extend(plot_per_seed(table, seeds, arguments.arms, arguments.output_dir, arguments.title))

  if arguments.replicate is not None:
    written.extend(plot_replicate(arguments.campaign, arguments.replicate, seeds, arguments.arms, arguments.output_dir))

  if arguments.replay is not None:
    replay_seeds = seed_directories([arguments.replay], ["meta"])
    written.extend(plot_replay(roots, arguments.replay, replay_seeds, arguments.output_dir))

  print("\nWROTE")
  for path in written:
    print(f"  {os.path.abspath(path)}")


if __name__ == "__main__":
  main()
