#!/usr/bin/env python3
"""How informative is an enzyme experiment? -- the landscape study behind the benchmark's retuning.

Scores designs with :mod:`detopt.bo.gbdt` (a GBDT stand-in for the meta-regressor, ~0.7 s a design
against its 27-137 s) and reports what the BO objective actually looks like. Three modes:

``scan``
    The informativeness PROFILE: a batch whose four experiments all sit at the same temperature,
    swept across ``temperature_bounds``. This is the diagnostic that exposes the benchmark's flaw --
    a design only learns anything while its temperatures sit near the melting prior, and the profile
    shows how wide that window is relative to the space BO searches.

``random``
    Landscape statistics over uniformly random designs: how much of the space is pinned at the
    no-information level (the target's own variance), how much headroom the rest has, and how that
    headroom compares with the evaluation noise. A benchmark whose best-so-far can keep improving
    for a hundred iterations needs a small plateau and many noise widths of graded range.

``repeat``
    Evaluation noise, measured rather than assumed: one design re-scored under independent event
    draws, against the same design re-scored under COMMON random numbers (a fixed event block).

``proxy``
    The justification for using a GBDT at all: re-scores designs a finished NEURAL campaign already
    evaluated and correlates the two, stratified over the neural loss. Run it under the config the
    campaign itself ran, NOT the current default -- a design scored under a different melting prior
    is a different quantity, and the correlation would be meaningless:

        python scripts/enzyme_landscape.py proxy --campaign output/enzyme \\
            --set enzyme.parameters.T_melting='[45.0, 58.0]'

Config knobs are overridden on the command line, so a sweep is a shell loop:

    python scripts/enzyme_landscape.py random --n-designs 256 \\
        --set enzyme.parameters.T_melting='[25.0, 75.0]'
"""

import argparse
import json
import os

import numpy as np

import detopt
import detopt.utils.config
from detopt.bo.gbdt import score_design

CONFIG = "config/detector/enzyme.yaml"


def load_detector(overrides):
  """The enzyme detector from its config, with dotted ``key=value`` overrides applied."""
  config = detopt.utils.config.override(detopt.utils.config.load_config(CONFIG), overrides)
  return detopt.detector.from_config(config), config


def random_designs(detector, rng, n):
  """``n`` designs uniform over the design bounds, in PHYSICAL (flat) form."""
  fraction = rng.uniform(*detector.enzyme_fraction_bounds, size=(n, detector.n_experiments))
  temperature = rng.uniform(*detector.temperature_bounds, size=(n, detector.n_experiments))
  return np.concatenate([fraction, temperature], axis=1).astype(np.float32)


def score_all(detector, designs, *, n_events, offsets, seed):
  """Score every design; ``offsets[i]`` picks the event block of design ``i``."""
  scores = []
  for design, offset in zip(designs, offsets):
    scores.append(score_design(detector, design, n_events=n_events, event_offset=int(offset), seed=seed))
  return scores


def physical(detector, scaled):
  """SCALED cube vector -> the flat PHYSICAL design the detector is called with.

  ``to_nominal`` returns the design as its named record, so it is flattened explicitly -- exactly
  what ``bo_gbdt.py`` does. Passing the record straight to ``np.asarray`` would build a (fields, m)
  array whose meaning depends on field order rather than a flat design vector."""
  return np.asarray(detector.flatten_design(detector.to_nominal(scaled)), dtype=np.float32)


def scaled_designs(detector, designs):
  """Physical flat designs -> the ``[0, 1]`` cube BO searches, where distances are comparable."""
  return np.stack([np.asarray(detector.to_scaled(d), dtype=np.float64) for d in designs])


def quotient_distance(a, b, detector):
  """Distance between two designs in the cube, MINIMISED over relabellings of the experiments.

  The batch is a set: two designs that differ only in the order of their experiments are the same
  design, and a plain Euclidean distance would call them far apart. A variogram built on the plain
  distance therefore reports scatter that is nothing but relabelling -- it would make an exactly
  symmetric landscape look rough. Enumerating the group is affordable while ``m! `` is small, which
  is the regime the benchmark lives in."""
  import itertools

  m = detector.n_experiments
  blocks = [np.arange(0, m), np.arange(m, 2 * m)]
  best = np.inf
  for order in itertools.permutations(range(m)):
    permuted = np.array(b, copy=True)
    for block in blocks:
      permuted[block] = b[block[list(order)]]
    best = min(best, float(np.linalg.norm(a - permuted)))
  return best


def best_of_k(losses, ks, rng, n_resample=4000):
  """``E[min of k draws]`` from the empirical landscape, for each ``k``.

  This is the random-search learning curve, and it is the honest answer to "does this benchmark
  reward more iterations?" WITHOUT running an optimiser: it depends only on the landscape's left
  tail. A benchmark whose best-of-120 matches its best-of-20 cannot reward a strategy that fits
  twice as many iterations into the same wall clock, however good the optimiser is."""
  losses = np.asarray(losses)
  return {
    int(k): float(np.mean(np.min(rng.choice(losses, size=(n_resample, int(k)), replace=True), axis=1)))
    for k in ks if k <= 4 * losses.size
  }


def variogram(detector, anchors, *, distances, n_events, seed, rng, celsius):
  """How fast does the loss change as a design is moved a given distance in the cube?

  Half the mean squared difference between pairs at separation ``h`` -- the semivariogram. It
  separates the two landscapes that a loss histogram cannot tell apart: one that varies gradually
  (gamma rising slowly, so a surrogate fitted at one point predicts its neighbours, and iterations
  compound) and one that is flat with a needle in it (gamma jumping to the full variance at the
  shortest separation, where no surrogate can help and the best strategy is to draw more points).

  The step is taken in the SCALED cube and clipped to it, so ``h`` is a fraction of the design range
  in every coordinate at once, and the realised distance is measured rather than assumed."""
  rows = []
  for anchor in anchors:
    base = score_design(detector, physical(detector, anchor), n_events=n_events, event_offset=0, seed=seed)
    for h in distances:
      step = rng.normal(size=anchor.size)
      step = step / np.linalg.norm(step) * h
      moved = np.clip(anchor + step, 0.0, 1.0)
      score = score_design(detector, physical(detector, moved), n_events=n_events, event_offset=0, seed=seed)
      rows.append({
        "requested": float(h),
        "distance": quotient_distance(anchor, moved, detector),
        "difference": float(score.loss - base.loss),
        # The same step in degrees of target resolution: sqrt of an MSE is an RMS, so the C figures
        # are differenced AFTER the square root, never before.
        "difference_c": float((np.sqrt(max(score.loss, 0.0)) - np.sqrt(max(base.loss, 0.0))) * celsius),
        "anchor_loss": float(base.loss)
      })
  summary = []
  for h in distances:
    group = [r for r in rows if r["requested"] == h]
    differences = np.array([r["difference"] for r in group])
    summary.append({
      "h": float(h),
      "n_pairs": len(group),
      "mean_distance": float(np.mean([r["distance"] for r in group])),
      "semivariance": float(0.5 * np.mean(differences ** 2)),
      "mean_abs_difference": float(np.mean(np.abs(differences))),
      "mean_abs_difference_c": float(np.mean(np.abs([r["difference_c"] for r in group])))
    })
  return summary, rows


def report_landscape(losses, sems, *, no_information):
  """The three numbers that decide whether BO iterations can pay off."""
  losses, sems = np.asarray(losses), np.asarray(sems)
  # "Plateau" = indistinguishable from knowing nothing: within 3 SEM of the target's own variance.
  plateau = losses > no_information - 3.0 * np.median(sems)
  best, median = float(losses.min()), float(np.median(losses))
  headroom = (median - best) / float(np.median(sems))
  return {
    "n": int(losses.size),
    "no_information": float(no_information),
    "plateau_fraction": float(plateau.mean()),
    "best": best,
    "p05": float(np.percentile(losses, 5)),
    "median": median,
    "sem_median": float(np.median(sems)),
    # How many noise widths separate a typical design from the best one -- the number of
    # DISTINGUISHABLE rungs BO has to climb, i.e. how much an extra iteration can still buy.
    "headroom_in_sem": float(headroom),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("mode", choices=["scan", "random", "screen", "floor", "repeat", "proxy"])
  parser.add_argument(
    "--campaign", default="output/enzyme", help="proxy: root of a finished neural campaign (<seed>/<strategy>/results.json)"
  )
  parser.add_argument("--n-events", type=int, default=16384, help="events sampled per design (generously)")
  parser.add_argument("--n-designs", type=int, default=256)
  parser.add_argument("--n-points", type=int, default=48, help="scan: temperatures swept")
  parser.add_argument("--n-repeats", type=int, default=24, help="repeat: independent re-scorings")
  parser.add_argument("--n-anchors", type=int, default=16, help="screen: designs the variogram steps away from")
  parser.add_argument("--fraction", type=float, default=0.3333333333333333, help="scan: the fixed enzyme fraction")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--common-random-numbers", action="store_true",
    help="score every design on the SAME event block (offset 0) instead of independent draws"
  )
  parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
  parser.add_argument("--output", default=None, help="write the raw scores here as JSON")
  arguments = parser.parse_args()

  detector, config = load_detector(arguments.overrides)
  rng = np.random.default_rng(arguments.seed)
  # The target is normalised to [-1, 1] over its own prior, so a uniform prior has variance 1/3 --
  # that is what a design carrying NO information scores.
  no_information = 1.0 / 3.0
  melting = config["enzyme"]["parameters"]["T_melting"]
  print(
    f"T_melting prior {melting}, temperature_bounds {list(detector.temperature_bounds)}, "
    f"n_experiments {detector.n_experiments}, noise {detector.measurement_noise}"
  )

  if arguments.mode == "scan":
    temperatures = np.linspace(*detector.temperature_bounds, arguments.n_points)
    designs = np.stack([
      np.concatenate([np.full(detector.n_experiments, arguments.fraction),
                      np.full(detector.n_experiments, t)]) for t in temperatures
    ]).astype(np.float32)
    scores = score_all(detector, designs, n_events=arguments.n_events, offsets=np.zeros(len(designs)), seed=arguments.seed)
    print(f"\n{'T (C)':>8} {'loss':>8} {'sem':>7} {'learners':>9}")
    for t, s in zip(temperatures, scores):
      bar = "#" * int(round(60 * (no_information - s.loss) / no_information))
      print(f"{t:8.1f} {s.loss:8.4f} {s.sem:7.4f} {s.n_learners:9d}  {bar}")
    losses = np.array([s.loss for s in scores])
    # "Informative" = the batch explains at least half the target's variance. A weaker test (any
    # significant drop below 1/3) also counts the cold tail, where the readout says a little about
    # the enzyme's speed but nothing useful about its melting point.
    informative = losses < 0.5 * no_information
    span = detector.temperature_bounds[1] - detector.temperature_bounds[0]
    print(
      f"\ninformative temperature window: {informative.mean() * span:.1f} C of {span:.0f} C "
      f"({informative.mean():.1%} of the design range); best {losses.min():.4f}"
    )
    # With n_experiments drawn independently over the range, this is the share of random designs
    # that learn nothing at all -- the plateau BO wastes its early iterations escaping.
    print(
      f"a random design misses that window with all {detector.n_experiments} experiments "
      f"{(1.0 - informative.mean()) ** detector.n_experiments:.1%} of the time"
    )
    payload = {"temperature": temperatures.tolist(), "loss": losses.tolist(), "informative_fraction": float(informative.mean())}

  elif arguments.mode == "random":
    designs = random_designs(detector, rng, arguments.n_designs)
    offsets = (np.zeros(len(designs), int) if arguments.common_random_numbers else arguments.n_events * np.arange(len(designs)))
    scores = score_all(detector, designs, n_events=arguments.n_events, offsets=offsets, seed=arguments.seed)
    losses = np.array([s.loss for s in scores])
    summary = report_landscape(losses, [s.sem for s in scores], no_information=no_information)
    print("\n" + json.dumps(summary, indent=2))
    counts, edges = np.histogram(losses, bins=20)
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
      print(f"  {low:.3f}-{high:.3f} {count:5d} {'#' * int(60 * count / max(counts.max(), 1))}")
    payload = {"summary": summary, "loss": losses.tolist(), "design": designs.tolist()}

  elif arguments.mode == "screen":
    # One variant of the benchmark, measured against the brief: SMOOTH, and REWARDS ITERATIONS.
    # Everything here is a property of the landscape, not of an optimiser, so it says whether a
    # setting is worth running BO on before any BO is run -- and it is ~200 evaluations rather than
    # ~1500. Common random numbers throughout: the objective is then a deterministic function of the
    # design, so a difference between two designs is the landscape and not the draw.
    half = 0.5 * (detector.melting_bounds[1] - detector.melting_bounds[0])
    designs = random_designs(detector, rng, arguments.n_designs)
    scores = score_all(detector, designs, n_events=arguments.n_events,
                       offsets=np.zeros(len(designs), int), seed=arguments.seed)
    losses = np.array([s.loss for s in scores])
    summary = report_landscape(losses, [s.sem for s in scores], no_information=no_information)

    curve = best_of_k(losses, (5, 10, 20, 40, 60, 120, 240), rng)
    scaled = scaled_designs(detector, designs)
    # Anchors: half uniform, half among the best draws. The uniform ones describe the space BO
    # spends its early iterations in; the good ones describe the basin it spends the rest in, which
    # is where smoothness decides whether iterations still buy anything.
    order = np.argsort(losses)
    n_anchor = max(2, arguments.n_anchors // 2)
    anchors = np.concatenate([scaled[order[:n_anchor]], scaled[rng.choice(len(scaled), n_anchor, replace=False)]])
    good = np.array([True] * n_anchor + [False] * n_anchor)
    # Out to 1.2, because two uniform points in this cube are ~1.15 apart: a ladder that stops short
    # of that cannot bracket the distance at which the loss decorrelates, and the correlation length
    # comes back undefined rather than large.
    ladder = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    profile, rows = variogram(detector, anchors, distances=ladder, n_events=arguments.n_events,
                              seed=arguments.seed, rng=rng, celsius=half)
    near_best, _ = variogram(detector, anchors[good], distances=ladder, n_events=arguments.n_events,
                             seed=arguments.seed, rng=np.random.default_rng(arguments.seed + 1), celsius=half)

    print("\n" + json.dumps(summary, indent=2))
    print(f"\nbest-of-k over random designs (does drawing more keep paying?)")
    print(f"{'k':>6} {'E[best]':>9} {'C RMSE':>8}   gain vs k=20")
    for k, value in curve.items():
      print(f"{k:6d} {value:9.4f} {np.sqrt(value) * half:8.2f}   "
            f"{curve.get(20, value) / value:6.3f}x")
    print(f"\nsemivariogram in the scaled cube (is it smooth, or flat with a needle?)")
    print(f"{'h':>6} {'|dloss|':>9} {'|dC|':>7} {'gamma':>9} {'gamma/var':>10}   (near-best anchors)")
    variance = float(np.var(losses))
    for row, close in zip(profile, near_best):
      print(f"{row['h']:6.2f} {row['mean_abs_difference']:9.4f} {row['mean_abs_difference_c']:7.2f} "
            f"{row['semivariance']:9.5f} {row['semivariance'] / variance:10.3f}   "
            f"{close['mean_abs_difference']:.4f} ({close['mean_abs_difference_c']:.2f} C)")
    print(f"\nlandscape variance {variance:.5f}; a gamma/var already ~1 at the shortest h is a "
          f"needle-in-a-plateau, and no surrogate can help on it.")

    # The verdict, as the two numbers the brief actually asks for.
    #
    # `correlation_length` is where the semivariance reaches half the landscape variance: how far a
    # design can move before its loss is unrelated to where it started, in units of the design range.
    # Big = smooth = a surrogate can extrapolate = iterations compound.
    #
    # `iteration_reward` is how much of the loss the last doubling of the budget still removes. A
    # benchmark that is already finished at 60 draws cannot pay a strategy back for reaching 120,
    # which is exactly the claim the four-strategy campaign has to demonstrate.
    #
    # `shotgun_headroom` is what is left for an optimiser at all: a 20-draw shotgun against the best
    # design seen. If it is ~1 there is nothing to win, however many iterations anyone spends.
    # The FIRST UP-CROSSING of gamma/var = 0.5, in units of the distance the step actually travelled.
    #
    # Two things this deliberately does NOT do. It does not interpolate against the REQUESTED ladder:
    # the step is clipped into the cube and then minimised over relabellings, so the realised
    # separation runs ~3-26% short and the requested value would inflate the answer. (The shortening
    # is roughly uniform across variants -- for a fixed step length the per-coordinate move is
    # h/sqrt(d), so a higher-dimensional cube clips LESS, not more.) And it does not sort by gamma
    # first: the profile is noisy and dips, but sorting rearranges the (gamma, distance) pairs out of
    # lag order and returns the 0.5-quantile of that rearrangement, which is not a correlation length
    # at all -- measured up to 4x wrong on a profile with a single dip past the crossing. A
    # variogram's crossing is the FIRST one; later excursions are noise about the sill.
    fractions = np.array([row["semivariance"] / variance for row in profile])
    heights = np.array([row["mean_distance"] for row in profile])
    above = np.flatnonzero(fractions >= 0.5)
    if above.size == 0:
      crossing = float("nan")  # never decorrelates over the measured ladder
    elif above[0] == 0:
      crossing = float(heights[0])  # already decorrelated at the shortest separation measured
    else:
      i = above[0]
      lo_f, hi_f = fractions[i - 1], fractions[i]
      span = hi_f - lo_f
      weight = 0.0 if span <= 0 else (0.5 - lo_f) / span
      crossing = float(heights[i - 1] + weight * (heights[i] - heights[i - 1]))
    #
    # The best-of-k estimates are BOOTSTRAP draws from the sample, so the sample's own minimum caps
    # them: at k comparable to `--n-designs` the curve flattens partly because it has run out of
    # sample, not because the landscape has run out of headroom. The bias is the same for every
    # variant, so it is safe to RANK on -- 20 -> 60 (k well inside the sample) is the figure to read,
    # and 60 -> 120 is the same statistic where the cap starts to bite.
    verdict = {
      "correlation_length": float(crossing),
      "iteration_reward_20_to_60": float(curve[20] / curve[60]) if 60 in curve else float("nan"),
      "iteration_reward_60_to_120": float(curve[60] / curve[120]) if 120 in curve else float("nan"),
      "shotgun_headroom_20": float(curve[20] / losses.min()),
      "best_c": float(np.sqrt(losses.min()) * half),
      "shotgun_20_c": float(np.sqrt(curve[20]) * half)
    }
    print(
      f"\nVERDICT  correlation length {verdict['correlation_length']:.2f} of the cube  |  "
      f"iteration reward 20->60 {verdict['iteration_reward_20_to_60']:.3f}x, "
      f"60->120 {verdict['iteration_reward_60_to_120']:.3f}x  |  "
      f"shotgun headroom {verdict['shotgun_headroom_20']:.2f}x "
      f"({verdict['shotgun_20_c']:.2f} C -> {verdict['best_c']:.2f} C)"
    )
    payload = {
      "verdict": verdict,
      "summary": summary,
      "best_of_k": curve,
      "best_of_k_c": {k: float(np.sqrt(v) * half) for k, v in curve.items()},
      "variogram": profile,
      "variogram_near_best": near_best,
      "variogram_rows": rows,
      "variance": variance,
      "loss": losses.tolist(),
      "design": designs.tolist()
    }

  elif arguments.mode == "floor":
    # The objective can NEVER reach zero: it is the average predictive MSE of a regressor recovering
    # T_melting from n_experiments x n_measurements NOISY reads, so it has an irreducible Bayes floor
    # set by `measurement_noise` and by the nuisance parameters drawn per enzyme. What a design
    # SCORES is that floor plus an estimation error that shrinks with the sample. Driving the sample
    # up separates the two: where the curve flattens is the floor, and only the distance ABOVE it is
    # anything a better estimator or more data could recover.
    designs = random_designs(detector, rng, arguments.n_designs)
    scored = score_all(detector, designs, n_events=arguments.n_events, offsets=np.zeros(len(designs)), seed=arguments.seed)
    design = designs[int(np.argmin([s.loss for s in scored]))]
    print(f"\nbest of {len(designs)} random designs, re-scored on a growing sample:")
    print(f"{'events':>9} {'loss':>8} {'train':>8} {'val':>8} {'sem':>7} {'learners':>9}")
    ladder, half = [], 0.5 * (detector.melting_bounds[1] - detector.melting_bounds[0])
    n = 4096
    while n <= arguments.n_events * 16:
      s = score_design(detector, design, n_events=n, event_offset=0, seed=arguments.seed)
      print(f"{n:9d} {s.loss:8.4f} {s.train:8.4f} {s.val:8.4f} {s.sem:7.4f} {s.n_learners:9d}"
            f"   ({np.sqrt(s.loss) * half:.2f} C RMSE)")
      ladder.append({"n_events": n, "loss": s.loss, "train": s.train, "val": s.val, "n_learners": s.n_learners})
      n *= 4
    print("\nwhere this flattens is the design's own Bayes floor; the target's variance is 1/3.")
    payload = {"design": design.tolist(), "ladder": ladder}

  elif arguments.mode == "proxy":
    # The claim the whole study rests on: a GBDT ranks designs the way the meta-regressor does.
    # Re-scores designs the finished neural campaign already evaluated, stratified over ITS loss so
    # the correlation is not just the plateau agreeing with itself.
    import glob as globbing

    from scipy.stats import pearsonr, spearmanr

    evaluated = []
    for path in sorted(globbing.glob(os.path.join(arguments.campaign, "*", "*", "results.json"))):
      with open(path) as f:
        evaluated.extend((np.asarray(r["design"], np.float32), float(r["loss"])) for r in json.load(f)["results"])
    if len(evaluated) == 0:
      raise SystemExit(f"no <seed>/<strategy>/results.json under {arguments.campaign}")
    neural = np.array([loss for _, loss in evaluated])
    order = np.argsort(neural)
    picked = order[np.linspace(0, order.size - 1, min(arguments.n_designs, order.size)).astype(int)]

    scores = [score_design(detector, evaluated[i][0], n_events=arguments.n_events, seed=arguments.seed) for i in picked]
    gbdt = np.array([s.loss for s in scores])
    neural = neural[picked]
    basin = neural < 0.2
    print(f"\n{len(picked)} designs of {len(evaluated)} re-scored at {arguments.n_events} events")
    print(f"neural loss {neural.min():.4f} .. {neural.max():.4f}; GBDT {gbdt.min():.4f} .. {gbdt.max():.4f}")
    print(f"Spearman {spearmanr(neural, gbdt).statistic:.4f}, Pearson {pearsonr(neural, gbdt)[0]:.4f}")
    print(
      f"inside the good basin (neural < 0.2, n={int(basin.sum())}): "
      f"Spearman {spearmanr(neural[basin], gbdt[basin]).statistic:.4f}"
    )
    payload = {
      "neural_loss": neural.tolist(),
      "gbdt_loss": gbdt.tolist(),
      "gbdt_sem": [s.sem for s in scores],
      "n_learners": [s.n_learners for s in scores]
    }

  else:  # repeat
    design = random_designs(detector, rng, 1)[0]
    # A design worth measuring noise on is an informative one; redraw until one is found.
    for _ in range(64):
      if score_design(detector, design, n_events=arguments.n_events, seed=arguments.seed).loss < 0.9 * no_information:
        break
      design = random_designs(detector, rng, 1)[0]
    independent = [
      score_design(detector, design, n_events=arguments.n_events, event_offset=k * arguments.n_events, seed=arguments.seed).loss
      for k in range(arguments.n_repeats)
    ]
    common = [
      score_design(detector, design, n_events=arguments.n_events, event_offset=0, seed=k).loss
      for k in range(arguments.n_repeats)
    ]
    print(f"\nindependent event draws: mean {np.mean(independent):.4f} sd {np.std(independent):.5f}")
    print(f"common random numbers  : mean {np.mean(common):.4f} sd {np.std(common):.5f}  (GBDT seed only)")
    payload = {"design": design.tolist(), "independent": independent, "common": common}

  if arguments.output is not None:
    os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
    # The settings the numbers were measured under travel WITH them. A landscape figure is about one
    # function; the config it came from is edited between studies, so a result that has to be read
    # against "the config" is a result that quietly changes meaning.
    payload["settings"] = {
      "mode": arguments.mode,
      "n_events": arguments.n_events,
      "n_experiments": int(detector.n_experiments),
      "n_measurements": int(detector.n_measurements),
      "temperature_bounds": [float(v) for v in detector.temperature_bounds],
      "enzyme_fraction_bounds": [float(v) for v in detector.enzyme_fraction_bounds],
      "melting_bounds": [float(v) for v in detector.melting_bounds],
      "measurement_noise": float(detector.measurement_noise),
      "duration": float(detector.duration),
      "concentration_E": float(detector.concentration_E),
      "half_time_bounds": [float(v) for v in detector.half_time_bounds],
      "celsius_per_unit": 0.5 * (float(detector.melting_bounds[1]) - float(detector.melting_bounds[0])),
      "overrides": list(arguments.overrides)
    }
    with open(arguments.output, "w") as f:
      json.dump(payload, f, indent=2)
    print(f"\nwrote {arguments.output}")


if __name__ == "__main__":
  main()
