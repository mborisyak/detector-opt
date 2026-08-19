#!/usr/bin/env python3
"""What the ``max_hits_per_event`` cap costs, split by what produced each hit.

Over capacity the C emitter keeps the M EARLIEST-TDC straws, so a hit survives or not by its TDC
rank within its event. One pass at the geometric-maximum buffer (nothing truncated) records every
fired straw with its TDC; each candidate cap is then applied host-side as a rank cut -- exactly the
emitter's rule -- so all caps come out of one simulation.

Three buckets, narrowing:

* ALL hits.
* ORIGINAL-PARTICLE hits -- kPPrimary, i.e. every particle the event file supplies (each boundary
  crossing), as opposed to the delta-rays / conversions / decay muons the tracker itself makes.
* HNL-DAUGHTER hits -- only the two ``product1``/``product2`` tracks of the truth record. The file's
  crossings are NOT just those two (an event's pool averages ~2.7 particles), and the solver tags a
  hit with the causing particle's PROCESS, not its identity, so the daughters are separated with a
  second solver pass: same pool, same row order, same per-event seed, but every non-daughter
  crossing started far outside the aperture so it fires nothing. Each primary's RNG key is the i-th
  split of the event key in pool-row order, so holding the rows in place leaves the daughters'
  trajectories bit-identical -- the straws they fire in that pass are exactly their straws in the
  full event. ``--check`` asserts this: on events whose pool IS just the two daughters, the pass
  must reproduce the primary bucket exactly.

    python scripts/hit_cap_loss.py --caps 384 128 96 --n-events 32768
"""

import argparse

import numpy as np

import detopt.detector
from detopt.detector.straw import Pool
from detopt.utils import config as config_utils
from detopt.utils.events import shuffled_event_index

PROC_PRIMARY = 0
# cm, added to a non-daughter crossing's y so it misses every straw (aperture half-height ~316 cm).
DISPLACEMENT = 1.0e5


def main(config_path, n_events, chunk, seed, caps, momentum_tolerance, check):
  config = config_utils.load_config(config_path)
  detector_config = config["detector"]
  name = next(iter(detector_config))
  configured_cap = int(detector_config[name]["max_hits_per_event"])
  detector_config[name]["max_hits_per_event"] = _geometric_max(detector_config, name)
  detector = detopt.detector.from_config(detector_config)
  design = config["nominal_design"]
  buffer_size = detector.max_hits_per_event

  index = shuffled_event_index(detector.size(), n_events, seed)
  print(f"detector={name} buffer={buffer_size} configured cap={configured_cap} n_events={n_events}")

  rank_limit = max(caps)
  fired = np.empty(n_events, dtype=np.int64)
  totals = {bucket: np.empty(n_events, dtype=np.int64) for bucket in ("original", "daughter")}
  by_rank = {bucket: np.zeros((n_events, rank_limit), dtype=bool) for bucket in ("original", "daughter")}
  clean = np.zeros(n_events, dtype=bool)
  disagreement = 0

  for start in range(0, n_events, chunk):
    stop = min(start + chunk, n_events)
    pool, boundaries, is_daughter_row, is_clean = _chunk_pool(detector, index[start:stop], momentum_tolerance)
    tdc, process_ids, straws = _solve(detector, design, index[start:stop], pool, boundaries, buffer_size)
    daughter_straws = _daughter_straws(detector, design, index[start:stop], pool, boundaries,
                                       is_daughter_row, buffer_size)

    valid = tdc >= 0.0
    original = (process_ids == PROC_PRIMARY) & valid
    daughter = np.take_along_axis(daughter_straws, np.where(valid, straws, 0), axis=1) & valid
    if check:
      disagreement += int((original[is_clean] != daughter[is_clean]).sum())

    order = np.argsort(np.where(valid, tdc, np.inf), axis=1, kind="stable")
    fired[start:stop] = valid.sum(axis=1)
    clean[start:stop] = is_clean
    for bucket, flags in (("original", original), ("daughter", daughter)):
      sorted_flags = np.take_along_axis(flags, order, axis=1)
      totals[bucket][start:stop] = sorted_flags.sum(axis=1)
      by_rank[bucket][start:stop] = sorted_flags[:, :rank_limit]
    print(f"  {stop}/{n_events}", end="\r", flush=True)
  print()

  if fired.max() >= buffer_size:
    raise SystemExit(f"buffer saturated at {buffer_size}: the counts are truncated")
  if check:
    print(f"[check] clean two-daughter events: {clean.sum()}/{n_events} ({100.0 * clean.mean():.1f}%); "
          f"daughter-vs-primary label disagreements on them: {disagreement}")
    if disagreement > 0:
      raise SystemExit("the displaced-crossing pass does not reproduce the primary bucket")

  print(f"\nper event: fired={fired.mean():.1f} original-particle={totals['original'].mean():.1f} "
        f"HNL-daughter={totals['daughter'].mean():.1f} (medians "
        f"{np.median(fired):.0f}/{np.median(totals['original']):.0f}/{np.median(totals['daughter']):.0f})")
  print(f"of all {fired.sum()} hits: {100.0 * totals['original'].sum() / fired.sum():.1f}% original-particle, "
        f"{100.0 * totals['daughter'].sum() / fired.sum():.1f}% HNL-daughter")

  report(caps, fired, totals, by_rank)


def _chunk_pool(detector, event_index, tolerance):
  """A compact ``Pool`` holding just this chunk's events, rows in their original per-event order, plus
  ``(boundaries, is_daughter_row, is_clean)``. Row order is preserved because a primary's RNG key is
  the i-th split of the event key."""
  events = detector._events
  offsets = events["offsets"]
  start, stop = offsets[event_index], offsets[event_index + 1]
  counts = (stop - start).astype(np.int64)
  rows = np.concatenate([np.arange(a, b) for a, b in zip(start, stop)]) if counts.sum() > 0 else np.zeros(0, int)
  boundaries = np.zeros((len(event_index), 2), dtype=np.int32)
  boundaries[:, 1] = np.cumsum(counts)
  boundaries[1:, 0] = boundaries[:-1, 1]
  pool = Pool(events["masses"][rows], events["charges"][rows], events["positions"][rows],
              events["momenta"][rows], events["times"][rows])

  # A crossing is a daughter when its momentum matches product1 or product2 of its event's truth.
  daughters = events["daughter_targets"][event_index] * 1000.0  # GeV/c -> MeV/c, the pool's unit
  per_row = np.repeat(np.arange(len(event_index)), counts)
  p1, p2 = daughters[per_row, 3:6], daughters[per_row, 6:9]
  is_daughter_row = (_distance(pool.momenta, p1) <= tolerance) | (_distance(pool.momenta, p2) <= tolerance)
  matched = np.bincount(per_row[is_daughter_row], minlength=len(event_index))
  is_clean = (counts == 2) & (matched == 2)
  return pool, boundaries, is_daughter_row, is_clean


def _solve(detector, design, event_index, pool, boundaries, buffer_size):
  """One solver call at the full buffer -> ``(tdc, process_ids, flat_straw_index)``, all ``(n, M)``.

  Uses the detector's own geometry map and per-index seeds, so these are the events ``__call__``
  produces; only the per-hit process codes, which ``__call__`` drops, need the direct engine call."""
  n = len(event_index)
  layers, angles, Bs = detector._design_to_geometry(detector._resolve_design(design, n))
  hits_idx = np.zeros((n, buffer_size, 4), dtype=np.uint32)
  tdc = np.full((n, buffer_size), -1.0, dtype=np.float32)
  process_ids = np.zeros((n, buffer_size), dtype=np.int32)
  detector.engine.solve(pool, np.ascontiguousarray(boundaries, np.int32), layers, angles, Bs,
                        hits_idx, tdc, seeds=detector._seeds(np.asarray(event_index, np.int64)),
                        process_ids=process_ids)
  station, view, lpv, straw = (hits_idx[..., i].astype(np.int64) for i in range(4))
  layer = (station * detector.n_views_per_station + view) * detector.n_layers_per_view + lpv
  return tdc, process_ids, layer * detector.n_straws + straw


def _daughter_straws(detector, design, event_index, pool, boundaries, is_daughter_row, buffer_size):
  """``(n, n_layers * n_straws)`` bool: the straws the two HNL daughters fire, from a re-solve of the
  same events with every non-daughter crossing displaced out of the aperture."""
  positions = pool.positions.copy()
  positions[~is_daughter_row, 1] += DISPLACEMENT
  displaced = Pool(pool.masses, pool.charges, positions, pool.momenta, pool.times)
  tdc, process_ids, straws = _solve(detector, design, event_index, displaced, boundaries, buffer_size)
  hit = (tdc >= 0.0) & (process_ids == PROC_PRIMARY)
  fired = np.zeros((len(event_index), detector.n_layers * detector.n_straws), dtype=bool)
  event_of_hit = np.broadcast_to(np.arange(len(event_index))[:, None], hit.shape)
  fired[event_of_hit[hit], straws[hit]] = True
  return fired


def _distance(a, b):
  return np.linalg.norm(a - b, axis=1)


def _geometric_max(detector_config, name):
  """Every straw in every layer -- an event cannot fire more (the emitter dedups per straw)."""
  arguments = detector_config[name]
  n_stations = int(arguments["n_stations_upstream"]) + int(arguments["n_stations_downstream"])
  n_views = int(arguments.get("n_views_per_station", 4))
  return n_stations * n_views * int(arguments["n_layers_per_view"]) * int(arguments["n_straws_per_layer"])


def report(caps, fired, totals, by_rank):
  n = len(fired)
  print(f"\n{'cap':>5}{'events over':>13}{'ALL kept':>11}"
        f"{'ORIGINAL kept':>15}{'lost':>8}{'ev hit':>9}"
        f"{'DAUGHTER kept':>15}{'lost':>8}{'ev hit':>9}")
  for cap in sorted(caps, reverse=True):
    line = f"{cap:>5}{100.0 * (fired > cap).mean():>12.2f}%{100.0 * np.minimum(fired, cap).sum() / fired.sum():>10.2f}%"
    for bucket in ("original", "daughter"):
      total = totals[bucket]
      kept = by_rank[bucket][:, :cap].sum(axis=1)
      lost = total - kept
      line += (f"{100.0 * kept.sum() / total.sum():>14.2f}%{100.0 * lost.sum() / total.sum():>7.2f}%"
               f"{100.0 * (lost > 0).mean():>8.2f}%")
    print(line)

  print(f"\nHNL daughters, within the {'events that lose any daughter hit':<40}")
  print(f"{'cap':>5}{'n events':>10}{'lost p50':>10}{'p90':>7}{'max':>7}"
        f"{'kept p50':>11}{'p10':>8}{'p01':>8}{'min':>8}")
  total = totals["daughter"]
  has = total > 0
  for cap in sorted(caps, reverse=True):
    kept = by_rank["daughter"][:, :cap].sum(axis=1)
    lost = total - kept
    fraction = np.divide(kept, total, out=np.ones(n), where=has)
    affected = lost > 0
    if affected.sum() == 0:
      print(f"{cap:>5}{0:>10}" + "".join(f"{'--':>{w}}" for w in (10, 7, 7, 11, 8, 8, 8)))
      continue
    f = 100.0 * fraction[affected]
    print(f"{cap:>5}{int(affected.sum()):>10}{np.percentile(lost[affected], 50):>10.0f}"
          f"{np.percentile(lost[affected], 90):>7.0f}{lost[affected].max():>7d}"
          f"{np.percentile(f, 50):>10.1f}%{np.percentile(f, 10):>7.1f}%{np.percentile(f, 1):>7.1f}%{f.min():>7.1f}%")

  thresholds = (1, 8, 16, 32)
  print(f"\nevents left with FEWER THAN k surviving HNL-daughter hits (uncapped baseline first)")
  print(f"{'cap':>5}" + "".join(f"{'k=' + str(k):>12}" for k in thresholds))
  for cap in [None] + sorted(caps, reverse=True):
    kept = total if cap is None else by_rank["daughter"][:, :cap].sum(axis=1)
    label = "none" if cap is None else str(cap)
    print(f"{label:>5}" + "".join(f"{int((kept < k).sum()):>7d}{100.0 * (kept < k).mean():>5.1f}%" for k in thresholds))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--config", default="config/bo.yaml")
  p.add_argument("--n-events", type=int, default=32 * 1024)
  p.add_argument("--chunk", type=int, default=512)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--caps", type=int, nargs="+", default=[384, 256, 192, 128, 96, 64])
  p.add_argument("--momentum-tolerance", type=float, default=1.0, help="MeV/c, for the daughter match")
  p.add_argument("--check", action="store_true", default=True)
  p.add_argument("--no-check", dest="check", action="store_false")
  a = p.parse_args()
  main(a.config, a.n_events, a.chunk, a.seed, a.caps, a.momentum_tolerance, a.check)
