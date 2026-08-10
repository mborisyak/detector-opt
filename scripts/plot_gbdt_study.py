#!/usr/bin/env python3
"""Figures for the enzyme landscape study (scripts/enzyme_landscape.py, scripts/bo_gbdt.py).

``landscape``
    Two panels: the informativeness profile (loss against a batch's common temperature) and the
    distribution of loss over random designs, for the baseline benchmark against a retuning. Both
    read the JSON those modes write, so the figure regenerates without re-scoring anything.

``convergence``
    Best-so-far loss against BO ITERATION, median over seeds, for BO against random search. Colour
    carries the benchmark variant and line style the search method, so neither is identified by
    colour alone.

``acceptance``
    The step-6 acceptance test as a JSON table + a printed summary: (1) does BO's best-so-far
    SEPARATE from the random null as iterations accumulate, and (2) does BO LOCALISE -- are its
    post-incumbent proposals closer to the incumbent than two independent uniform points are to
    each other? The second is the one that distinguishes a GP that has learned the landscape from
    one that is guessing, which a loss curve alone cannot.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("AGG")

import numpy as np
from matplotlib.figure import Figure

# The repo's chart palette (detopt/utils/viz/bo.py). Blue and orange are the two most reliably
# separable hues under every common colour-vision deficiency, so the two-variant comparisons use
# them, and line style repeats the distinction for print and forced-colour rendering.
BASE_COLOUR, TUNED_COLOUR = "#2a78d6", "#eb6834"
NO_INFORMATION = 1.0 / 3.0


def _load(path):
  with open(path) as f:
    return json.load(f)


def _settings(path):
  """The PROBLEM settings a run was made under -- everything except the surrogate.

  ``kernel`` is dropped because it is a property of the optimiser, not of the function being
  optimised: a random-search arm never touches it, so one null is valid for every kernel arm over
  the same problem. Everything else (the admissible box, the batch size, the enzyme population, the
  read-out noise, the normaliser) moves the objective itself, and two runs that differ in any of it
  are measurements of different functions."""
  payload = _load(path)
  settings = dict(payload.get("settings") or {})
  settings.pop("kernel", None)
  # The SEARCH SPACE is part of the function being optimised, and it is recorded at the payload's
  # top level rather than inside `settings`. A uniform draw over the shared-fraction diagonal is not
  # the same null as a uniform draw over the full cube, so an arm may not borrow across it.
  for key in ("searched_dimensions", "shared_fraction", "fixed_fraction", "n_events"):
    settings[key] = payload.get(key)
  # `settings` lists only the fields the driver thought to record. `overrides` is the command line
  # that produced the run, so it catches everything else the objective depends on -- a different
  # kinetic prior, concentration or calibration -- which otherwise passes the guard unseen.
  settings["overrides"] = tuple(payload.get("overrides") or ())
  return settings


def _kernel_name(path):
  kernel = (_load(path).get("settings") or {}).get("kernel")
  return next(iter(kernel), "?") if isinstance(kernel, dict) else "?"


def _describe(path):
  """A label that states the settings, so no figure can outlive the config it was measured under."""
  data = _load(path)
  settings = data.get("settings")
  if settings is None:
    return f"({os.path.basename(path)}: no recorded settings)"
  low, high = settings["temperature_bounds"]
  return (f"T in [{low:g}, {high:g}] C, m={settings['n_experiments']} experiments, "
          f"noise={settings['measurement_noise']:g}, T_melt prior {settings['melting_bounds']}")


def _style(axis):
  axis.grid(True, alpha=0.25, lw=0.6)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)


def landscape(scans, randoms, output, labels):
  figure = Figure(figsize=(11, 4.4))
  profile, distribution = figure.subplots(1, 2)

  for (name, path), colour in zip(scans.items(), (BASE_COLOUR, TUNED_COLOUR)):
    data = _load(path)
    profile.plot(
      data["temperature"], data["loss"], lw=2.0, color=colour, marker="o", ms=4,
      label=f"{labels.get(name, name)}  ({data['informative_fraction']:.0%} informative)"
    )
  profile.axhline(NO_INFORMATION, color="#666666", lw=1.2, ls=":")
  profile.text(
    0.99, NO_INFORMATION, "no information (target variance)", ha="right", va="bottom", transform=profile.get_yaxis_transform(),
    fontsize=8, color="#444444"
  )
  profile.set_xlabel("temperature of every experiment in the batch (C)")
  profile.set_ylabel("GBDT loss (normalised MSE)")
  profile.set_title("Where an experiment is informative")
  profile.legend(fontsize=8, loc="lower left")
  _style(profile)

  bins = np.linspace(0.0, 0.36, 37)
  for (name, path), colour in zip(randoms.items(), (BASE_COLOUR, TUNED_COLOUR)):
    losses = np.array(_load(path)["loss"])
    distribution.hist(
      losses, bins=bins, color=colour, alpha=0.55, label=f"{labels.get(name, name)}  (median {np.median(losses):.3f})"
    )
  distribution.axvline(NO_INFORMATION, color="#666666", lw=1.2, ls=":")
  distribution.set_xlabel("GBDT loss of a uniformly random design")
  distribution.set_ylabel("designs")
  distribution.set_title("What a random design is worth")
  distribution.legend(fontsize=8, loc="upper left")
  _style(distribution)

  figure.tight_layout()
  figure.savefig(output, dpi=130)
  print(f"wrote {output}")


def _curves(paths, key):
  """Per-seed best-so-far arrays, truncated to the shortest run."""
  runs = []
  for path in sorted(paths):
    results = _load(path)["results"]
    if len(results) > 0 and key not in results[0]:
      raise SystemExit(f"{os.path.basename(path)} has no {key!r} -- it was run with --no-verify, so "
                       f"there is no held-out curve to plot")
    values = np.array([r[key] for r in results], dtype=np.float64)
    if key.startswith("verified"):
      # Score the design the OPTIMISED loss selected, rather than taking a running minimum of the
      # held-out values -- that would be selecting on the held-out block, which is the one thing it
      # exists to prevent (measured up to 4.2% optimistic).
      chosen = np.minimum.accumulate(np.array([r["loss"] for r in results], dtype=np.float64))
      picked = np.array([int(np.argmin([r["loss"] for r in results][:i + 1])) for i in range(len(results))])
      runs.append(values[picked])
    else:
      runs.append(np.minimum.accumulate(values))
  if len(runs) == 0:
    return None
  length = min(len(r) for r in runs)
  return np.stack([r[:length] for r in runs])


def _localisation(path):
  """Is the optimiser LOCALISING or guessing?

  For every proposal after the first, the L2 distance in the SCALED cube from the incumbent at that
  time, divided by the mean distance between two independent uniform points in the same cube. A GP
  that has learned something proposes NEAR its incumbent, so the ratio drops well below 1.

  Read it against the RANDOM arm's own ratio, not against 1.0. A uniform proposal is not expected to
  score exactly 1.0: the incumbent is usually interior (good designs are), and the mean distance from
  a uniform point to a fixed interior point is a little SHORTER than between two uniform points. The
  null arm measures that guessing baseline directly under the same geometry, which is what makes it
  the honest comparison."""
  results = _load(path)["results"]
  x = np.asarray([r["x_scaled"] for r in results], dtype=np.float64)
  loss = np.asarray([r["loss"] for r in results], dtype=np.float64)
  if x.shape[0] < 3:
    return None
  # The mean distance between two uniform points in [0, 1]^d, by Monte Carlo at the run's own d.
  rng = np.random.default_rng(0)
  d = x.shape[1]
  reference = float(np.mean(np.linalg.norm(rng.random((20000, d)) - rng.random((20000, d)), axis=1)))
  ratios = []
  for i in range(1, x.shape[0]):
    best = int(np.argmin(loss[:i]))  # the incumbent the proposal was made against
    ratios.append(float(np.linalg.norm(x[i] - x[best])) / reference)
  return np.asarray(ratios)


def _variants(directory):
  """Arm names, recovered by stripping the trailing `-<mode>-<seed>` -- NOT `split("-")[0]`, which
  silently drops every arm whose own name contains a hyphen."""
  names = set()
  for path in glob.glob(os.path.join(directory, "*-*-*.json")):
    stem = os.path.basename(path)[:-len(".json")]
    arm, _, rest = stem.rpartition("-")      # drop seed
    arm, _, mode = arm.rpartition("-")       # drop mode
    if mode in ("bo", "random") and rest.isdigit() and len(arm) > 0:
      names.add(arm)
  return sorted(names)


def _paths(directory, variant, mode):
  return sorted(glob.glob(os.path.join(directory, f"{variant}-{mode}-*.json")))


def _summary(paths, key="loss"):
  stacked = _curves(paths, key)
  if stacked is None:
    return None
  median = np.median(stacked, axis=0)
  celsius = (_load(paths[0]).get("settings") or {}).get("celsius_per_unit")
  entry = {
    "n_runs": int(stacked.shape[0]),
    "n_iterations": int(median.size),
    "median_best_at": {str(k): float(median[k - 1]) for k in (5, 10, 20, 40, 60, 80, 120) if k <= median.size},
    "final_median": float(median[-1]),
    "final_spread": [float(stacked[:, -1].min()), float(stacked[:, -1].max())]
  }
  if celsius is not None:
    # The same medians as a target resolution in C -- the only figure comparable across variants
    # that change the prior, the population or the objective.
    entry["median_rmse_c_at"] = {k: float(np.sqrt(max(v, 0.0)) * celsius) for k, v in entry["median_best_at"].items()}
  ratios = [r for r in (_localisation(p) for p in paths) if r is not None]
  if len(ratios) > 0:
    length = min(r.size for r in ratios)
    stack = np.stack([r[:length] for r in ratios])
    entry["localisation_ratio_all"] = float(np.median(stack))
    entry["localisation_ratio_second_half"] = float(np.median(stack[:, length // 2:]))
  return entry


def acceptance(directory, output, null_variant):
  """The step-6 acceptance test, as a JSON table: does BO separate from the null, and does it
  localise? Both are read off the run JSONs, so nothing is re-scored.

  A variant with no random arm of its own is compared against ``null_variant`` -- and only after its
  PROBLEM settings are checked to be identical, because a null measured on a different function is
  not a null. Arms that differ only in the surrogate kernel legitimately share one."""
  report, shared = {}, _paths(directory, null_variant, "random")
  for variant in _variants(directory):
    arm, bo_paths = {}, _paths(directory, variant, "bo")
    null_paths = _paths(directory, variant, "random")
    if len(null_paths) == 0 and len(bo_paths) > 0 and len(shared) > 0:
      mismatch = [p for p in bo_paths if _settings(p) != _settings(shared[0])]
      if len(mismatch) > 0:
        mine, theirs = _settings(mismatch[0]), _settings(shared[0])
        differing = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
        detail = "\n".join(f"    {k}: {mine.get(k)!r} vs {theirs.get(k)!r}" for k in differing)
        raise SystemExit(
          f"{variant!r} has no random arm and cannot borrow {null_variant!r}: they are not the same "
          f"problem.\n  {os.path.basename(mismatch[0])} vs {os.path.basename(shared[0])} differ in:\n"
          f"{detail}"
        )
      null_paths, arm["null_from"] = shared, null_variant
    for mode, paths in (("bo", bo_paths), ("random", null_paths)):
      entry = _summary(paths)
      if entry is not None:
        arm[mode] = entry
    if len(arm) == 0:
      continue
    if len(bo_paths) > 0:
      arm["kernel"] = _kernel_name(bo_paths[0])
      arm["settings"] = _describe(bo_paths[0])
    elif len(null_paths) > 0:
      arm["settings"] = _describe(null_paths[0])
    if "bo" in arm and "random" in arm:
      # Separation: how much lower BO's median best-so-far is than the null's, per iteration count.
      arm["bo_over_null"] = {
        k: float(arm["random"]["median_best_at"][k] / arm["bo"]["median_best_at"][k])
        for k in arm["bo"]["median_best_at"] if k in arm["random"]["median_best_at"]
      }
    report[variant] = arm

  if len(report) == 0:
    raise SystemExit(f"no run JSON matching <variant>-<mode>-<seed>.json under {directory}")
  with open(output, "w") as f:
    json.dump(report, f, indent=2)

  for variant, arm in report.items():
    print(f"\n=== {variant} ===  kernel={arm.get('kernel', '-')}  {arm.get('settings', '')}")
    for mode in ("bo", "random"):
      if mode not in arm:
        continue
      e = arm[mode]
      at = "  ".join(f"@{k}={v:.4f}" for k, v in e["median_best_at"].items())
      borrowed = "  (shared null)" if mode == "random" and "null_from" in arm else ""
      print(f"  {mode:<7} n={e['n_runs']:<3} {at}   spread@end=[{e['final_spread'][0]:.4f}, "
            f"{e['final_spread'][1]:.4f}]{borrowed}")
      if "median_rmse_c_at" in e:
        print("          in C:  " + "  ".join(f"@{k}={v:.2f}" for k, v in e["median_rmse_c_at"].items()))
      if "localisation_ratio_all" in e:
        print(
          f"          localisation (prop->incumbent / 2 random): all={e['localisation_ratio_all']:.3f} "
          f"second-half={e['localisation_ratio_second_half']:.3f}"
        )
    if "bo_over_null" in arm:
      print("  null/BO: " + "  ".join(f"@{k}={v:.2f}x" for k, v in arm["bo_over_null"].items()))
  print(f"\nwrote {output}")


def screens(directory):
  """Rank the variants screened by ``enzyme_landscape.py screen`` (one JSON each) on the brief.

  The brief is a TRADE-OFF, so the table is deliberately not sorted on one column: a variant wants a
  long correlation length (smooth enough that a surrogate helps) AND headroom left over a shotgun
  (something for the extra iterations to win). Either alone is easy and useless -- a flat landscape
  is perfectly smooth, and a needle has unlimited headroom no optimiser can reach."""
  rows = []
  for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
    data = _load(path)
    if "verdict" not in data:
      continue
    settings = data.get("settings", {})
    rows.append({
      "variant": os.path.basename(path)[:-5],
      "m": settings.get("n_experiments"),
      "box": settings.get("temperature_bounds"),
      "prior": settings.get("melting_bounds"),
      "noise": settings.get("measurement_noise"),
      "plateau": data["summary"]["plateau_fraction"],
      "median_c": float(np.sqrt(data["summary"]["median"]) * settings["celsius_per_unit"]),
      **data["verdict"]
    })
  if len(rows) == 0:
    raise SystemExit(f"no screen JSON (with a `verdict`) under {directory}")

  header = (f"{'variant':<14} {'m':>2} {'box':>12} {'prior':>12} {'noise':>6} {'plat':>5} "
            f"{'corr.len':>8} {'20->60':>7} {'60->120':>8} {'headroom':>9} {'shotgun C':>10} {'best C':>7}")
  print(header)
  print("-" * len(header))
  for row in sorted(rows, key=lambda r: -r["shotgun_headroom_20"]):
    box = f"[{row['box'][0]:g},{row['box'][1]:g}]" if row["box"] else "?"
    prior = f"[{row['prior'][0]:g},{row['prior'][1]:g}]" if row["prior"] else "?"
    print(f"{row['variant']:<14} {row['m']:>2} {box:>12} {prior:>12} {row['noise']:>6.3f} "
          f"{row['plateau']:>5.2f} {row['correlation_length']:>8.2f} "
          f"{row['iteration_reward_20_to_60']:>7.3f} {row['iteration_reward_60_to_120']:>8.3f} "
          f"{row['shotgun_headroom_20']:>9.2f} {row['shotgun_20_c']:>10.2f} {row['best_c']:>7.2f}")
  print("\ncorr.len: distance (in design ranges) over which the loss decorrelates -- larger is smoother.")
  print("20->60 / 60->120: how much of the loss another doubling of random draws still removes.")
  print("headroom: a 20-draw shotgun against the best design seen; ~1 means nothing left to optimise.")
  print("C columns are the physical target resolution, comparable across variants that move the prior.")


def convergence(directory, output, key):
  """Best-so-far against iteration, one colour per ARM found in the directory (no hardcoded arm
  names -- a figure that names a box the runs were not made in is worse than no figure). The
  subtitle states the settings every arm shares, and an arm that differs is labelled with its own."""
  figure = Figure(figsize=(8.0, 5.2))
  axis = figure.subplots()
  variants = _variants(directory)
  palette = [BASE_COLOUR, TUNED_COLOUR, "#3f9e5a", "#8a4fbd", "#c2352b", "#6b7280"]
  described, found = set(), False
  for variant, colour in zip(variants, palette * len(variants)):
    for mode, style, marker in (("bo", "-", "o"), ("random", "--", "s")):
      paths = _paths(directory, variant, mode)
      stacked = _curves(paths, key)
      if stacked is None:
        continue
      found = True
      described.add(_describe(paths[0]))
      median = np.median(stacked, axis=0)
      iterations = np.arange(1, median.size + 1)
      axis.plot(
        iterations, median, style, color=colour, lw=2.0, marker=marker, ms=4, markevery=6,
        label=f"{variant} — {'BO' if mode == 'bo' else 'random search'} ({stacked.shape[0]} seeds)"
      )
      if stacked.shape[0] > 1:
        axis.fill_between(iterations, stacked.min(axis=0), stacked.max(axis=0), color=colour, alpha=0.12, lw=0)
  if not found:
    raise SystemExit(f"no run JSON matching <variant>-<mode>-<seed>.json under {directory}")
  axis.set_xlabel("BO iteration")
  axis.set_ylabel(f"best-so-far loss ({'held-out' if key.startswith('verified') else 'optimised'})")
  axis.set_yscale("log")
  # The settings line sits between the title and the axes, so the title needs padding for it --
  # without that they are drawn at the same height and overprint each other.
  axis.set_title("Does the benchmark reward more iterations?", pad=20 if len(described) == 1 else 6)
  if len(described) == 1:
    axis.text(0.5, 1.005, described.pop(), transform=axis.transAxes, ha="center", va="bottom", fontsize=8,
              color="#444444")
  axis.legend(fontsize=8)
  _style(axis)
  figure.tight_layout()
  figure.savefig(output, dpi=130)
  print(f"wrote {output}")


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="mode", required=True)

  one = sub.add_parser("landscape")
  one.add_argument("--scan", action="append", required=True, metavar="NAME=PATH")
  one.add_argument("--random", action="append", required=True, metavar="NAME=PATH")
  one.add_argument("--output", required=True)

  two = sub.add_parser("convergence")
  two.add_argument("--directory", required=True, help="holds <variant>-<mode>-<seed>.json from bo_gbdt.py")
  two.add_argument("--output", required=True)
  two.add_argument("--key", default="loss", choices=["loss", "verified_loss"])

  three = sub.add_parser("acceptance")
  three.add_argument("--directory", required=True, help="holds <variant>-<mode>-<seed>.json from bo_gbdt.py")
  three.add_argument("--output", required=True, help="the JSON table to write")
  three.add_argument(
    "--null", default="null",
    help="variant whose random arm serves any BO arm without one (only if the problem settings match)"
  )

  four = sub.add_parser("screens")
  four.add_argument("--directory", required=True, help="holds one JSON per `enzyme_landscape.py screen` variant")

  arguments = parser.parse_args()
  if arguments.mode == "screens":
    screens(arguments.directory)
  elif arguments.mode == "landscape":
    pairs = lambda items: dict(item.split("=", 1) for item in items)
    landscape(pairs(arguments.scan), pairs(arguments.random), arguments.output, {})
  elif arguments.mode == "acceptance":
    acceptance(arguments.directory, arguments.output, arguments.null)
  else:
    convergence(arguments.directory, arguments.output, arguments.key)


if __name__ == "__main__":
  main()
