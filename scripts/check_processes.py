"""Per-process hits/event comparison: StrawDetector simulation vs FairShip MC truth.

For a sample of events we tally, per physics process, the number of straw hits each
event produces, and overlay the simulated distribution on the MC-truth one. Each hit
is bucketed into a common category:

    primary    an INCOMING particle (its track originated upstream of the tracker,
               origin z < boundary_z). The sim is fed every boundary crossing and
               tracks them all as proc 0, so on the MC side ALL upstream-origin tracks
               (true daughters AND upstream secondaries that crossed in) are primary --
               otherwise the MC would split them by their upstream proc and the
               per-event primary multiplicity would not be comparable.
    pair       proc 5, produced in the tracker (gamma -> e+e-).
    decay      proc 4, in the tracker (pi/K -> mu decay-in-flight).
    secondary  in-tracker delta-rays / other procs + MC untracked soft shower
               (hit_track == -2); SIM delta-rays (9) + noise (31).

The sim models the MC's untracked soft-secondary band with explicit delta-rays/pairs,
so the comparison is distributional (per-event multiplicity), not hit-by-hit.

    python scripts/check_processes.py [npz_path_or_dir] [n_events]

A directory pools every *.npz it contains (same ordering/offset as the event loader).
Writes output/check_processes.png and prints a composition summary.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import yaml  # noqa: E402

from detopt.detector.free_straw import FreeStrawDetector, free_design_array  # noqa: E402

CATS = ["primary", "pair", "decay", "secondary"]
# TMCProcess code -> category. Unlisted codes fall through to "secondary".
# ship2numpy mc_particles column holding each track's origin-vertex z (cm).
MC_ORIGIN_Z_COL = 7


def _nominal_design(det, cfg_path="config/detector/nominal_design.yaml"):
    nd = yaml.safe_load(open(cfg_path))["nominal_design"]
    return free_design_array(
        nd["station_z"],
        n_layers_per_view=det.n_layers_per_view,
        view_angles=nd["view_angles"],
        view_z_gap=nd["view_z_gap"],
        layer_z_gap=nd["layer_z_gap"],
        B=nd["B"],
    )


def _resolve_files(npz):
    """File list for a path, matching load_ship2numpy_events: a dir pools its sorted
    *.npz; a file loads itself; a bare dir-stem falls back to ship2numpy.npz."""
    path = Path(npz)
    if path.is_dir():
        files = sorted(path.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no .npz files in directory: {path}")
        return files
    if path.is_file():
        return [path]
    if (path / "ship2numpy.npz").exists():
        return [path / "ship2numpy.npz"]
    raise FileNotFoundError(f"event file not found: {path}")


def load_mc_hits(npz):
    """Concatenated MC-truth hit arrays across one or many files, with per-file event
    indices offset by the cumulative ``truth`` row count -- the SAME offsetting the event
    loader applies, so MC event indices line up with the detector's pooled events.

    Returns ``(hit_track, hit_event_index, mc_info, mc_event_index, origin_z)`` (globally indexed).
    """
    ht, hev, mci, mcev, oz = [], [], [], [], []
    ev_off = 0
    for f in _resolve_files(npz):
        d = np.load(f, allow_pickle=True)
        n_ev = int(np.asarray(d["truth"]).shape[0])
        mco = np.asarray(d["mc_info"], np.int64)
        mp = np.asarray(d["mc_particles"], np.float32)
        ht.append(np.asarray(d["hit_track"], np.int64))
        hev.append(np.asarray(d["hit_event_index"], np.int64) + ev_off)
        mci.append(mco)
        mcev.append(np.asarray(d["mc_event_index"], np.int64) + ev_off)
        # per-track origin z (cm); some files lack the origin column -> NaN (proc-based fallback)
        if mp.ndim == 2 and mp.shape[1] > MC_ORIGIN_Z_COL and mp.shape[0] == mco.shape[0]:
            oz.append(mp[:, MC_ORIGIN_Z_COL])
        else:
            print(f"  note: {f.name} mc_particles has no origin column (shape {mp.shape}) -> proc-based fallback")
            oz.append(np.full(mco.shape[0], np.nan, np.float32))
        ev_off += n_ev
    return (np.concatenate(ht), np.concatenate(hev), np.concatenate(mci), np.concatenate(mcev), np.concatenate(oz))


def _cat_index(codes):
    """Map an array of TMCProcess codes to category indices into CATS."""
    idx = np.full(len(codes), CATS.index("secondary"), dtype=np.int64)
    for code, cat in PROC2CAT.items():
        idx[codes == code] = CATS.index(cat)
    return idx


def mc_hit_categories(ht, hev, mci, mcev, origin_z, boundary_z, n_events):
    """Per-MC-hit category index, vectorised.

    ``hit_track`` is the per-event-local MC track id (== mc_info row within the event);
    ``-2`` marks untracked soft-shower hits. For each tracked hit, resolve its global
    mc row, then bucket: any track whose ORIGIN is upstream of the tracker
    (``origin_z < boundary_z``) is an incoming particle -> ``primary`` (matching how the
    sim labels every boundary crossing); tracks produced IN the tracker keep their proc
    (5 -> pair, 4 -> decay, else secondary). Untracked (-2) -> secondary.
    """
    mc_start = np.searchsorted(mcev, np.arange(n_events + 1))  # event -> first mc row
    cat = np.full(len(ht), CATS.index("secondary"), dtype=np.int64)
    tracked = ht >= 0
    g = mc_start[hev[tracked]] + ht[tracked]
    proc, oz = mci[g, 3], origin_z[g]
    c = np.full(len(g), CATS.index("secondary"), dtype=np.int64)
    c[proc == 5] = CATS.index("pair")  # in-tracker conversion
    c[proc == 4] = CATS.index("decay")  # in-tracker decay
    c[proc == 0] = CATS.index("primary")  # true primaries (robust if origin z is missing/NaN)
    c[oz < boundary_z] = CATS.index("primary")  # upstream origin -> incoming particle
    cat[tracked] = c
    return cat


def main():
    npz = sys.argv[1] if len(sys.argv) > 1 else "ship2numpy.npz"
    n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 400

    cfg = yaml.safe_load(open("config/detector/straw.yaml"))["straw"]
    det = FreeStrawDetector(**cfg, data_dir=npz)
    design = _nominal_design(det)[None, :]
    events = det._events
    o = events["offsets"]
    rng = np.random.default_rng(0)

    ht, hev, mci, mcev, origin_z = load_mc_hits(npz)
    mc_cat = mc_hit_categories(ht, hev, mci, mcev, origin_z, det.boundary_z, det.n_events)
    have_hits = set(np.unique(hev).tolist())

    cand = [e for e in range(det.n_events) if e in have_hits and int(o[e + 1]) - int(o[e]) > 0]
    show = np.asarray(cand[:n_show], dtype=np.int64)
    n_ev = len(show)

    # --- SIM: batched solve in CHUNKS, with only the per-hit process_ids buffer (no
    # trajectories / MC tree). Chunking bounds peak memory (a single 100k-event solve
    # would allocate a ~GB X/mask it discards) while keeping the batched C throughput.
    # Vectorised TMCProcess-code -> category lookup (codes are small; 31 noise -> secondary).
    lut = np.full(64, CATS.index("secondary"), dtype=np.int64)
    for code, ci in {0: 0, 5: 1, 4: 2, 9: 3}.items():
        lut[code] = ci
    M = det.max_hits_per_event
    CHUNK = 8000
    sim_mat = np.zeros((n_ev, len(CATS)), dtype=np.int64)
    for st in range(0, n_ev, CHUNK):
        sl = show[st : st + CHUNK]
        bnd = np.stack([o[sl], o[sl + 1]], axis=1).astype(np.int32)
        design_b = np.ascontiguousarray(np.broadcast_to(design, (len(sl), design.shape[1])))
        pid = np.zeros((len(sl), M), dtype=np.int32)
        _, mk, _ = det._run_solver(bnd, design_b, rng, process_ids=pid)
        mk = mk.astype(bool)
        sidx = lut[np.clip(pid, 0, 63)]
        for ci in range(len(CATS)):
            sim_mat[st : st + len(sl), ci] = ((sidx == ci) & mk).sum(1)
        print(f"  solved {min(st + CHUNK, n_ev)}/{n_ev} events", flush=True)

    # --- MC: one scatter-add to tally (event, category), then select the shown events.
    mc_all = np.zeros((det.n_events, len(CATS)), dtype=np.int64)
    np.add.at(mc_all, (hev, mc_cat), 1)
    mc_mat = mc_all[show]

    sim_counts = {c: sim_mat[:, ci] for ci, c in enumerate(CATS)}
    mc_counts = {c: mc_mat[:, ci] for ci, c in enumerate(CATS)}

    # ---- composition summary ----------------------------------------------------
    sim_tot = sum(sim_counts[c].sum() for c in CATS)
    mc_tot = sum(mc_counts[c].sum() for c in CATS)
    print(f"events: {n_ev}   total hits  sim {sim_tot}  MC {mc_tot}")
    print(f"{'process':>10} | {'sim hits/ev':>12} {'sim %':>7} | {'MC hits/ev':>11} {'MC %':>7}")
    for c in CATS:
        s, m = sim_counts[c], mc_counts[c]
        print(
            f"{c:>10} | {s.mean():12.2f} {100*s.sum()/max(sim_tot,1):6.1f}% | "
            f"{m.mean():11.2f} {100*m.sum()/max(mc_tot,1):6.1f}%"
        )

    # ---- per-process histograms -------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, c in zip(axes.ravel(), CATS):
        s, m = sim_counts[c], mc_counts[c]
        # Upper limit clamped to min(max, 1.5 * q95) so fat delta-ray tails don't
        # compress the bulk; combine sim+MC for a shared, comparable x-range.
        allv = np.concatenate([s, m])
        hi = int(min(allv.max(initial=0), 1.25 * np.percentile(allv, 98))) if len(allv) else 1
        hi = max(hi, 1)
        bins = np.linspace(0, hi, min(hi, 40) + 1)
        ax.hist(m, bins=bins, color="0.4", alpha=0.55, label=f"MC (mean {m.mean():.1f})")
        ax.hist(s, bins=bins, histtype="step", color="crimson", lw=1.8, label=f"sim (mean {s.mean():.1f})")
        ax.set_xlim(0, hi)
        ax.set_title(f"{c} hits / event")
        ax.set_xlabel("hits per event")
        ax.set_ylabel("events")
        ax.legend()
    src = Path(npz).name if Path(npz).is_file() else f"{Path(npz).name}/ ({len(_resolve_files(npz))} files)"
    fig.suptitle(f"hits/event by process: sim vs MC  ({n_ev} events, {src})")
    fig.tight_layout()
    out_path = Path("output/check_processes.png")
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=110)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
