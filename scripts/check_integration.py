"""Integration-error check: adaptive dt vs a tiny fixed-dt ground truth.

Runs the same data-file events through (a) a fine fixed step dt=DT_REF (ground
truth) and (b) the adaptive stepper at several max_dt. All stochastic processes
are disabled (no secondaries / conversion / noise) so the only difference is the
trajectory integration. We compare the set of fired straws (a hit is the tuple
(station,view,layer,straw)) and the TDC time on the hits both agree on.

    python scripts/check_integration.py [npz_path] [n_events]
"""

import sys

import numpy as np

import yaml

from detopt.detector.free_straw import FreeStrawDetector, free_design_array


def _nominal_design(det, cfg_path="config/detector/straw.yaml"):
    nd = yaml.safe_load(open(cfg_path))["nominal_design"]
    return free_design_array(
        nd["station_z"], n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"], view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"], B=nd["B"],
    )

NPZ = sys.argv[1] if len(sys.argv) > 1 else "ship2numpy.npz"
N_EVENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
DT_REF = 0.01  # ns, fine fixed-step ground truth

# Disable every stochastic channel so only the integrator differs.
OFF = dict(wall_thickness=0.0, lambda_conv_cm=0.0, noise_rate=0.0, enable_decay=False, max_particles=25)


def _event_dd(events, e):
    """Single-event daughter_data sliced from the sparse pool (flat + offsets)."""
    o = events["offsets"]
    s = slice(int(o[e]), int(o[e + 1]))
    n = int(o[e + 1]) - int(o[e])
    return {
        "masses": events["masses"][s],
        "charges": events["charges"][s],
        "positions": events["positions"][s],
        "momenta": events["momenta"][s],
        "times": events["times"][s],
        "offsets": np.array([0, n], dtype=np.int32),
    }


def _event_nparticles(events, e):
    o = events["offsets"]
    return int(o[e + 1]) - int(o[e])


def hitset_and_times(det, events, idx, seed):
    """Return {(st,vw,lv,straw): time} over `idx` events for one detector."""
    design = _nominal_design(det)[None, :]
    out = {}
    for e in idx:
        dd = _event_dd(events, e)
        X, mask, _ = det._run_solver(dd, design, np.random.default_rng(seed))
        h = X[0, mask[0].astype(bool)]
        for row in h:
            out[(e, int(row[0]), int(row[1]), int(row[2]), int(row[3]))] = float(row[4])
    return out


def main():
    ref = FreeStrawDetector(data_dir=NPZ, dt=DT_REF, max_time=300.0, **OFF)
    events = ref._events
    idx = [e for e in range(ref.n_events) if _event_nparticles(events, e) > 0][:N_EVENTS]
    truth = hitset_and_times(ref, events, idx, seed=0)
    print(f"ground truth: fixed dt={DT_REF} ns, {len(idx)} events, {len(truth)} hits")
    print("(time RMS is over hits both agree on; it is TDC-smearing-dominated, ~flat in max_dt)\n")
    print(f"{'max_dt':>7} {'recovered':>10} {'spurious':>9} {'time RMS(ns)':>12}")

    for max_dt in (0.25, 0.5, 1.0, 2.0, 5.0):
        det = FreeStrawDetector(data_dir=NPZ, max_dt=max_dt, max_time=300.0, **OFF)
        got = hitset_and_times(det, events, idx, seed=0)
        gk, tk = set(got), set(truth)
        inter = gk & tk
        recovered = len(inter) / max(len(tk), 1)
        spurious = len(gk - tk) / max(len(tk), 1)
        dts = np.array([got[k] - truth[k] for k in inter])
        trms = np.sqrt(np.mean(dts**2)) if len(dts) else 0.0
        print(f"{max_dt:7.2f} {recovered*100:9.2f}% {spurious*100:8.2f}% {trms:12.4f}")


if __name__ == "__main__":
    main()
