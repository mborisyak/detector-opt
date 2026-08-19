#!/usr/bin/env python3
"""Hit-multiplicity distribution of a straw detector at one fixed design.

Simulates ``--n-events`` events at the config's ``nominal_design`` and reports the per-event
fired-straw count as quantiles. The buffer is sized to ``n_layers * n_straws`` -- the geometric
maximum -- so the count is the UNTRUNCATED number of distinct straws the solver fired; the
config's own ``max_hits_per_event`` cap is then applied host-side as ``min(count, M)``, which is
exactly what the C emitter does (over capacity it keeps the M earliest-TDC straws), so both the
true and the NN-facing distribution come out of a single pass.

    python scripts/hit_multiplicity.py --config config/bo.yaml --n-events 32768
"""

import argparse

import numpy as np

import detopt.detector
from detopt.utils import config as config_utils
from detopt.utils.events import shuffled_event_index

QUANTILES = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]


def main(config_path, n_events, chunk, seed, save=None):
  config = config_utils.load_config(config_path)
  detector_config = config["detector"]
  name = next(iter(detector_config))
  cap = int(detector_config[name]["max_hits_per_event"])

  # Size the emit buffer to the geometric maximum so nothing is truncated. Build the detector once
  # at the configured cap to read the geometry, then rebuild at the full size -- the event pool load
  # dominates, so read the geometry off a no-data twin instead.
  detector_config[name]["max_hits_per_event"] = _geometric_max(detector_config, name)
  detector = detopt.detector.from_config(detector_config)
  design = config["nominal_design"]

  size = detector.size()
  print(f"detector={name} events available={size} design={design}")
  print(f"buffer max_hits_per_event={detector.max_hits_per_event} (geometric max), config cap={cap}")

  index = shuffled_event_index(size, n_events, seed)
  counts = np.empty(n_events, dtype=np.int64)
  for start in range(0, n_events, chunk):
    stop = min(start + chunk, n_events)
    _ground_truth, _event, mask, _target = detector(design, index[start:stop])
    counts[start:stop] = np.asarray(mask).sum(axis=1)
    print(f"  {stop}/{n_events}", end="\r", flush=True)
  print()

  if counts.max() >= detector.max_hits_per_event:
    raise SystemExit(f"buffer saturated at {detector.max_hits_per_event}: the counts are truncated")

  report("untruncated (all fired straws)", counts)
  report(f"as configured (max_hits_per_event={cap})", np.minimum(counts, cap))
  over = int((counts > cap).sum())
  empty = int((counts == 0).sum())
  print(f"\nevents over the {cap} cap: {over}/{n_events} ({100.0 * over / n_events:.2f}%)")
  print(f"events with no hits at all:  {empty}/{n_events} ({100.0 * empty / n_events:.2f}%)")
  if save is not None:
    np.savez(save, counts=counts, event_index=index, cap=cap)
    print(f"counts -> {save}")
  return counts


def _geometric_max(detector_config, name):
  """Every straw in every layer -- an event cannot fire more (the emitter dedups per straw)."""
  arguments = detector_config[name]
  n_stations = int(arguments["n_stations_upstream"]) + int(arguments["n_stations_downstream"])
  n_views = int(arguments.get("n_views_per_station", 4))
  return n_stations * n_views * int(arguments["n_layers_per_view"]) * int(arguments["n_straws_per_layer"])


def report(title, counts):
  quantiles = np.quantile(counts, QUANTILES, method="linear")
  print(f"\n{title}: n={len(counts)} mean={counts.mean():.1f} std={counts.std():.1f}")
  print("  " + "  ".join(f"{label:>7}" for label in ("min", "p50", "p80", "p90", "p95", "max")))
  print("  " + "  ".join(f"{value:>7.0f}" for value in quantiles))


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("--config", default="config/bo.yaml")
  p.add_argument("--n-events", type=int, default=32 * 1024)
  p.add_argument("--chunk", type=int, default=512)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--save", default=None, help="npz destination for the per-event counts")
  a = p.parse_args()
  main(a.config, a.n_events, a.chunk, a.seed, a.save)
