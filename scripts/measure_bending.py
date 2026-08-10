"""Push the two HNL daughters through the stereo straw detector and histogram their field bending.

Each daughter is deflected by the spectrometer's magnetic field. From the shared decay vertex and
the daughter's first detector-plane crossing we define the INITIAL direction ``n0`` (essentially
straight -- the field is weak upstream of the first station); the FINAL direction ``n1`` is the
trajectory tangent at the end (chord between the last two planes it crosses). This script reports
two per-daughter bending observables:

  * ``angle(n0, n1)`` -- the total deflection angle across the magnet, and
  * bending ``Δy`` -- the y displacement at the last plane relative to the no-field straight line
    (vertex -> first plane, extrapolated to ``z_last``). The bend is essentially all in y (the field
    is along x), so this is the physical sagitta the spectrometer measures.

It samples events at the config design, asks the C solver for each daughter's ground-truth ``(x, y)``
crossing of every layer plane (``_simulate(..., z_planes=layer_z, n_tracks=2)``), plots both
histograms, and prints summary statistics.

    python scripts/measure_bending.py seed=0 n_events=4096

This touches only the detector's existing public interface -- it does not modify any detector.
"""

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import detopt  # noqa: E402
from detopt.utils.events import shuffled_event_index  # noqa: E402


def measure(seed=0, n_events=4096, out="bending.png", **config):
    detector = detopt.detector.from_config(config["detector"])

    # Nominal (un-scaled) design from the config `design:` ref -> per-layer z planes.
    theta = detector.to_scaled(config["design"])
    phys = np.asarray(detector.flatten_design(detector.to_nominal(theta)), np.float32)
    layer_z = np.asarray(detector._design_to_geometry(phys[None, :])[0][0], np.float32)  # (n_planes,)
    design = np.broadcast_to(phys[None, :], (int(n_events), phys.shape[0]))  # one row per event

    # Push the two daughters through; tracks[e, d, k] = (x, y) crossing of plane k by daughter d.
    event_index = shuffled_event_index(detector.size(), int(n_events), int(seed))
    out_ev = detector._simulate(design, event_index, z_planes=layer_z, n_tracks=2)
    traj, ncr = out_ev["traj"], out_ev["n_cross"]  # (n,2,P,3) ordered (x,y,z) crossings, (n,2) counts
    n, _, P, _ = traj.shape
    lz = np.asarray(layer_z, np.float32)
    tracks = np.zeros((n, 2, P, 2), np.float32)  # per-plane (x,y); scatter ordered crossings back by z
    crossed = np.zeros((n, 2, P), bool)
    vc = np.arange(P)[None, None, :] < ncr[:, :, None]
    pp = np.clip(np.searchsorted(lz, traj[..., 2]), 0, P - 1)
    ei = np.broadcast_to(np.arange(n)[:, None, None], (n, 2, P))[vc]
    es = np.broadcast_to(np.arange(2)[None, :, None], (n, 2, P))[vc]
    pv = pp[vc]
    tracks[ei, es, pv, 0] = traj[..., 0][vc]
    tracks[ei, es, pv, 1] = traj[..., 1][vc]
    crossed[ei, es, pv] = True
    vertex = np.asarray(out_ev["target"].vertex)[:, None, :]  # (n, 1, 3) shared HNL decay vertex (x, y, z), lab cm

    n, m, P, _ = tracks.shape
    z = np.broadcast_to(layer_z[None, None, :, None], (n, m, P, 1))
    xyz = np.concatenate([tracks, z], axis=-1)  # (n, m, n_planes, 3) full-3D crossings

    # First / last / second-last crossed plane per daughter (planes are z-ordered).
    idx = np.arange(P)
    first = np.where(crossed, idx, P).min(-1)  # (n, m)
    last = np.where(crossed, idx, -1).max(-1)
    second_last = np.where(crossed & (idx != last[..., None]), idx, -1).max(-1)
    valid = crossed.sum(-1) >= 2  # need >=2 crossings to define both an initial and a final segment

    clamp = lambda k: np.clip(k, 0, P - 1)  # non-crossing daughters hold out-of-range sentinels; filtered by `valid`
    first, last, second_last = clamp(first), clamp(last), clamp(second_last)
    take = lambda src, k: np.take_along_axis(src, k[..., None, None], axis=2)[:, :, 0, :]  # (n, m, 3)
    p_first, p_last, p_prev = take(xyz, first), take(xyz, last), take(xyz, second_last)

    n0 = p_first - vertex  # initial direction: decay vertex -> first plane
    n1 = p_last - p_prev  # final direction: tangent at the end of the trajectory
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    cos = np.sum(unit(n0) * unit(n1), axis=-1)  # (n, m)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))[valid]  # pooled per-daughter bending

    # Bending Δy: y at the last plane minus the no-field straight line (vertex -> first plane)
    # extrapolated to z_last. n0_y / n0_z is the initial y-slope; the residual is the field sagitta.
    slope_y = n0[..., 1] / n0[..., 2]
    y_straight = vertex[..., 1] + slope_y * (p_last[..., 2] - vertex[..., 2])
    delta_y = (p_last[..., 1] - y_straight)[valid]  # (n_daughters,) cm

    print(f"design layers (z): {np.array2string(layer_z, precision=1)}")
    print(f"events={n_events}  daughters with >=2 plane crossings: {angle.size}")
    print(
        f"bending angle [deg]:  mean={angle.mean():.3f}  median={np.median(angle):.3f}  "
        f"p90={np.percentile(angle, 90):.3f}  p99={np.percentile(angle, 99):.3f}  max={angle.max():.3f}"
    )
    print(
        f"bending Δy   [cm] :  mean={delta_y.mean():+.3f}  std={delta_y.std():.3f}  "
        f"median|Δy|={np.median(np.abs(delta_y)):.3f}  p99|Δy|={np.percentile(np.abs(delta_y), 99):.3f}"
    )

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5))
    hi = np.percentile(angle, 99)  # clip the soft-daughter tail so the bulk is visible
    ax0.hist(np.clip(angle, 0.0, hi), bins=120, color="C3", alpha=0.85)
    ax0.set_xlabel(r"bending angle $\angle(n_0, n_1)$  [deg]")
    ax0.set_ylabel("daughters")
    ax0.set_title("deflection angle")

    yhi = np.percentile(np.abs(delta_y), 99)
    ax1.hist(np.clip(delta_y, -yhi, yhi), bins=120, color="C0", alpha=0.85)
    ax1.set_xlabel(r"bending $\Delta y = y_{\mathrm{last}} - y_{\mathrm{straight}}$  [cm]")
    ax1.set_ylabel("daughters")
    ax1.axvline(0.0, color="k", lw=0.8, ls="--")
    ax1.set_title("y sagitta")

    fig.suptitle(f"HNL daughter field bending in stereo detector ($N$={angle.size})")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")
    return angle, delta_y


if __name__ == "__main__":
    import gearup

    gearup.gearup(measure).with_config("config/regression.yaml")()
