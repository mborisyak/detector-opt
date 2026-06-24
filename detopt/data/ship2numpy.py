"""Preloader for the ``ship2numpy`` boundary-crossing event file.

A pure function: read the ``.npz`` once into a flat, sparse per-event pool and
hand it back. No train/val splits, no iteration partitions, no mutable state
-- that machinery does not belong to the event source (the detector just gathers
from the preloaded pool by ``event_index``; the scripts own any split).

The pool is *sparse* (CSR-style): particles are stored as flat arrays ordered by
event, with an ``offsets`` array delimiting each event's slice. No padding to
``max_particles`` -- the C solver indexes the flat arrays via the offsets.
"""

from pathlib import Path

import numpy as np

__all__ = ["load_ship2numpy_events"]


def load_ship2numpy_events(path, boundary_z=None):
    """Read a ``ship2numpy`` ``.npz`` into a flat, sparse per-event pool.

    Every boundary crossing is kept -- the flat (CSR) layout carries variable-length
    events, so there is no per-event cap (a busy event keeps all its daughters).

    File arrays used (MC-truth ``mc_*``/``hits`` are ignored):

        truth (N, 15): HNL[mass, p(3), decay_vertex(3)], product1[mass, p(3)],
                       product2[mass, p(3)]  -- mass GeV, p GeV/c, vertex cm.
        particles (M, >=8): [charge(e), mass(GeV), px, py, pz (GeV/c),
                       x, y (cm @ boundary), t0 (ns)].
        event_index (M,) int32: row into ``truth`` for each particle.
        boundary_z scalar: z of the crossing plane (used when ``boundary_z`` arg
                       is None).

    Returns a dict with the flat per-particle arrays (units match the C solver:
    masses MeV, momenta MeV/c, positions cm with ``z = boundary_z``, times ns),
    ordered by event, plus ``offsets (N+1,)`` delimiting each event's slice (each
    event capped at ``max_particles``); the 6-vector ``targets`` ``[decay_vertex(cm),
    HNL momentum(GeV)]``, the resolved ``boundary_z``, and ``n_events``.
    """
    # Resolve to a list of .npz files: a directory pre-loads ALL its files
    # (concatenated into one pool); a single file loads just that one.
    path = Path(path)
    if path.is_dir():
        npz_paths = sorted(path.glob("*.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"no .npz event files in directory: {path}")
    elif path.is_file():
        npz_paths = [path]
    elif (path / "ship2numpy.npz").exists():  # back-compat: dir stem
        npz_paths = [path / "ship2numpy.npz"]
    else:
        raise FileNotFoundError(f"ship2numpy file not found: {path}")

    # Concatenate every file, offsetting each file's event_index into the global
    # truth so events from different files keep distinct indices.
    truths, parts, eidx = [], [], []
    boundary_z_file, ev_offset = None, 0
    for p in npz_paths:
        data = np.load(p)
        t = np.asarray(data["truth"], dtype=np.float32)
        pa = np.asarray(data["particles"], dtype=np.float32)
        ei = np.asarray(data["event_index"], dtype=np.int64)
        if pa.shape[0] != ei.shape[0]:
            raise ValueError(f"{p}: particles ({pa.shape[0]}) and event_index ({ei.shape[0]}) lengths disagree")
        if pa.shape[1] < 8:
            raise ValueError(
                f"{p}: particles has {pa.shape[1]} columns; the signed-charge layout "
                f"[charge, mass, px, py, pz, x, y, t0] (>=8 columns) is required."
            )
        truths.append(t)
        parts.append(pa)
        eidx.append(ei + ev_offset)
        ev_offset += t.shape[0]
        if boundary_z_file is None and "boundary_z" in data:
            boundary_z_file = float(np.asarray(data["boundary_z"]).item())
    truth = np.concatenate(truths, axis=0)
    particles = np.concatenate(parts, axis=0)
    event_index = np.concatenate(eidx, axis=0)

    if boundary_z is None:
        boundary_z = boundary_z_file if boundary_z_file is not None else 8403.0
    boundary_z = float(boundary_z)

    # Current (only) layout: [charge, mass, px, py, pz, x, y, t0].
    charge = particles[:, 0].astype(np.float32)
    mass_gev = particles[:, 1]
    mom_gev = particles[:, 2:5]
    cross_x = particles[:, 5]
    cross_y = particles[:, 6]
    t0 = particles[:, 7]

    n_events = int(truth.shape[0])

    # Keep EVERY valid crossing. A stable sort by event_index groups each event's
    # particles contiguously (ascending event order), so the kept rows are already in
    # flat event order and per-event counts give the CSR offsets. No cap.
    order = np.argsort(event_index, kind="stable")
    kept = order[(event_index[order] >= 0) & (event_index[order] < n_events)]
    counts = np.bincount(event_index[kept], minlength=n_events).astype(np.int64)

    masses = (mass_gev[kept] * 1000.0).astype(np.float32)  # GeV -> MeV
    charges = charge[kept].astype(np.float32)
    positions = np.empty((kept.shape[0], 3), dtype=np.float32)
    positions[:, 0] = cross_x[kept]
    positions[:, 1] = cross_y[kept]
    positions[:, 2] = boundary_z
    momenta = (mom_gev[kept] * 1000.0).astype(np.float32)  # GeV/c -> MeV/c
    times = t0[kept].astype(np.float32)
    offsets = np.zeros(n_events + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)

    targets = np.concatenate([truth[:, 4:7], truth[:, 1:4]], axis=1).astype(np.float32)
    # Per-event HNL kinematics for conditioning: [mass, px, py, pz] (GeV). Mass (truth[:,0])
    # is constant in single-mass samples but kept per-event for future multi-mass runs.
    conditioning = truth[:, 0:4].astype(np.float32)
    # DAUGHTER target [decay_vertex(3), product1_p(3), product2_p(3)] (GeV) -- for the
    # tracking detector that predicts the two daughters' momenta instead of the HNL's.
    daughter_targets = truth[:, [4, 5, 6, 8, 9, 10, 12, 13, 14]].astype(np.float32)

    return {
        "masses": masses,
        "charges": charges,
        "positions": positions,
        "momenta": momenta,
        "times": times,
        "offsets": offsets,
        "targets": targets,
        "daughter_targets": daughter_targets,
        "conditioning": conditioning,
        "boundary_z": boundary_z,
        "n_events": n_events,
    }
