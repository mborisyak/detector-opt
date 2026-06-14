"""Fire the straw detector and save a few visualisations.

New Detector contract (detector-spec.md): events come from
``detector.sample_events(seed, design)`` ->
``{ground_truth, X, mask, targets, trajectories}`` where ``X`` is the dense
padded ``(B, M, 5)`` hit array ``[station, view, layer_in_view, straw, time]``
and ``mask`` marks the real hits. No sparse containers, no splits, no state.

Run headless:  ``python scripts/viz.py [ship2numpy.npz] [out_dir]``
"""

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never block
import matplotlib.pyplot as plt
import numpy as np

import detopt


def _build_detector(data_path):
    return detopt.detector.FreeStrawDetector(
        n_stations=4,
        n_views_per_station=4,
        n_layers_per_view=2,
        n_straws_per_layer=200,
        straw_pitch=2.0,
        straw_length=400.0,
        max_B=0.15,
        B_sigma=300.0,
        z0=8957.0,
        layer_bounds=(8200.0, 9750.0),
        max_particles=64,
        data_dir=data_path,
        # the three "easy" physics processes, on so the plots show their effect
        lambda_conv_cm=50.0,
        enable_decay=True,
        noise_rate=3.0,
    )


def _hit_world(detector, X, mask, design):
    """Map dense hits to world coords for the event display.

    Returns ``(z, y, station)`` arrays for the masked hits of one event:
    the hit's layer z (from the design) and its straw transverse position.
    """
    m = mask.astype(bool)
    station = X[m, 0].astype(int)
    view = X[m, 1].astype(int)
    layer_in_view = X[m, 2].astype(int)
    straw = X[m, 3]
    per_station = detector.n_views_per_station * detector.n_layers_per_view
    layer = station * per_station + view * detector.n_layers_per_view + layer_in_view
    positions = np.asarray(design[: detector.n_layers], dtype=np.float32)
    z = positions[np.clip(layer, 0, detector.n_layers - 1)]
    y_stagger = np.where(layer_in_view & 1, 0.5 * detector.layer_y_offset, -0.5 * detector.layer_y_offset)
    y = (straw + 0.5) * detector.straw_pitch - detector.layer_height + y_stagger
    return z, y, station


def _event_display(detector, out, X, mask, trajectories, design, n_events=3):
    positions = np.asarray(design[: detector.n_layers], dtype=np.float32)
    for ev in range(min(n_events, X.shape[0])):
        if mask[ev].sum() == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        # layer planes
        for zc in positions:
            ax.axvline(zc, color="0.85", lw=0.6, zorder=0)
        # hits, coloured by station
        z, y, station = _hit_world(detector, X[ev], mask[ev], design)
        sc = ax.scatter(z, y, c=station, cmap="tab10", vmin=0, vmax=9, s=10, zorder=3)
        # daughter trajectories (z vs y), drop padded (all-zero) steps
        traj = trajectories[ev]  # (max_particles, n_t, 3)
        for p in range(traj.shape[0]):
            pts = traj[p]
            live = np.abs(pts).sum(axis=1) > 0
            if live.sum() > 1:
                ax.plot(pts[live, 2], pts[live, 1], lw=0.7, alpha=0.6, zorder=2)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("transverse y [cm]")
        ax.set_title(f"Event {ev}: {int(mask[ev].sum())} hits")
        fig.colorbar(sc, ax=ax, label="station")
        fig.tight_layout()
        path = os.path.join(out, f"viz_event_{ev}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  wrote {path}")


def _tdc_histogram(out, X, mask):
    times = X[..., 4][mask.astype(bool)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.asarray(times, dtype=float), bins=60, alpha=0.8)
    ax.set_xlabel("TDC time [ns]")
    ax.set_ylabel("counts")
    ax.set_title(f"FairShip-style TDC times ({times.size} hits)")
    fig.tight_layout()
    path = os.path.join(out, "viz_tdc.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main(data_path="ship2numpy.npz", out="output/viz", seed=123, batch=256):
    os.makedirs(out, exist_ok=True)
    detector = _build_detector(data_path)
    print(f"detector: {detector.n_events} source events, boundary_z={detector.boundary_z} cm")

    import yaml

    nd = yaml.safe_load(open("config/detector/straw.yaml"))["nominal_design"]
    design = detopt.detector.free_design_array(
        nd["station_z"], n_layers_per_view=detector.n_layers_per_view,
        view_angles=nd["view_angles"], view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"], B=nd["B"],
    )
    designs = np.tile(design[None, :], (batch, 1)).astype(np.float32)

    out_dict = detector.sample_events(seed, designs)
    X, mask = np.asarray(out_dict["X"]), np.asarray(out_dict["mask"])
    trajectories = np.asarray(out_dict["trajectories"])
    print(f"generated {batch} events, mean {mask.sum(1).mean():.1f} hits/event, " f"{int((mask.sum(1) > 0).sum())} with hits")

    _event_display(detector, out, X, mask, trajectories, design)
    _tdc_histogram(out, X, mask)

    # quick throughput number
    t0 = time.perf_counter()
    detector(seed + 1, designs)
    dt = time.perf_counter() - t0
    print(f"throughput: {batch / dt:.0f} events/s ({dt * 1e3:.0f} ms for {batch} events)")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "ship2numpy.npz"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/viz"
    main(data, out_dir)
