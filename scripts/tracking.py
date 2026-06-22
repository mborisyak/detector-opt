"""FairShip verification: our own 2-track trackers (retina + NLL/MAP, ``detopt.tracking``) over raw
``StrawEvent`` records of the ``StereoTracking`` sim, across the four measurement targets and both cut
regimes, compared to FairShip. ``report`` writes report.txt with two master tables (physical units + R^2),
each split into four regime blocks (material OFF/ON x our-cuts/FairShip-cuts); ``track`` runs one config.

Targets: V1 tubes (wire centre), V2 TDC (raw ToF+drift+propagation -- the network's proxy), V3 r_drift
(smeared drift radius -- FairShip-comparable), V4 hits (exact x,y -- the sanity floor).

    python scripts/tracking.py report
    python scripts/tracking.py track config/tracking/nll-tdc.yaml material=False n_files=2
"""
import numpy as np
import yaml
from tabulate import tabulate

import detopt
import detopt.tracking as tracking
from detopt.detector.straw import StrawEvent
import oursim_retina as O  # _load_reco / _input_events / _no_material (data pipeline reuse)
import fairship_hits  # third regime: our trackers on FairShip's own hits (real field map)

DETECTOR_CFG = "config/detector/stereo.yaml"
DESIGN_CFG = "config/design/initial_stereo.yaml"

# FairShip ShipAna / Ship2NumPy selection (scripts/fairship_select.py): per daughter track >= 25 straw
# measurements, >= 3 of 4 stations crossed, fitted chi2/Ndf < 4, and a 2-track DoCA <= 2 cm. "Our cuts"
# are modelled after these but computed from OUR fit; "FairShip cuts" = the events FairShip reconstructed.
MEAS_CUT, MIN_STATIONS, CHI2_CUT, DOCA_CUT = 25, 3, 4.0, 2.0
PHYS_N_TRACKS = 8  # material-ON: the 2 daughters + up to 6 secondary tracks as unlabeled noise hits

METHODS = ["retina-tubes", "retina-tdc", "retina-drift", "retina-hits",
           "nll-tubes", "nll-tdc", "nll-drift", "nll-hits"]
CODE = {"retina-tubes": "R-Tu", "retina-tdc": "R-TD", "retina-drift": "R-Dr", "retina-hits": "R-Hi",
        "nll-tubes": "N-Tu", "nll-tdc": "N-TD", "nll-drift": "N-Dr", "nll-hits": "N-Hi"}


def _slice_event(event, i, j):
    """Slice every leaf of a tracking ``StrawEvent`` over the event axis (None leaves stay None)."""
    sl = lambda a: None if a is None else a[i:j]
    return StrawEvent(station=event.station[i:j], view=event.view[i:j], layer=event.layer[i:j],
                      straw=event.straw[i:j], tdc=event.tdc[i:j], x=sl(event.x), y=sl(event.y),
                      drift_r=sl(event.drift_r))


def _solver_trajectory(det, ie, bnds, design, layer_z, n_tracks, primaries, batch=4096):
    """Solve the selected events, returning the per-track trajectory (``traj``, ``n_cross``) AND the
    digitised hits (packed ``StrawEvent`` ``digi`` + ``digi_mask``) -- the latter carries the real TDC
    that V2 looks up by straw address. ``primaries=False`` also records secondaries as noise tracks."""
    n, m = bnds.shape[0], layer_z.shape[0]
    traj = np.zeros((n, n_tracks, m, 3), np.float32)
    ncr = np.zeros((n, n_tracks), np.int32)
    Xs, masks = [], []
    rng = np.random.default_rng(0)
    for i in range(0, n, batch):
        b = bnds[i:i + batch]
        nb = b.shape[0]
        tj, nc = np.zeros((nb, n_tracks, m, 3), np.float32), np.zeros((nb, n_tracks), np.int32)
        pi = np.full((nb, n_tracks), -1, np.int32)
        X, mk, _ = det._run_solver(b, np.repeat(design[None], nb, 0), rng, input_events=ie, z_planes=layer_z,
                                   traj=tj, n_cross=nc, part_idx=pi, primaries=primaries)
        traj[i:i + nb], ncr[i:i + nb] = tj, nc
        Xs.append(np.asarray(X))
        masks.append(np.asarray(mk))
    return traj, ncr, det._pack_event(np.concatenate(Xs, 0)), np.concatenate(masks, 0)


def load_tracking_data(det, design, n_files, n_events, material, n_tracks, primaries):
    """Truth + FairShip reco + the tracking ``StrawEvent`` (with x, y, smeared drift_r, and the real TDC)
    for the first ``n_events`` events. ``primaries=False`` (material ON) records secondaries into the
    extra slots. Returns ``(event, mask, true9, reco9, reco_ok)`` -- the last three on the first
    ``n_events`` events (9 = [vertex(3) cm, p1(3) GeV, p2(3) GeV])."""
    if not material:
        O._no_material(det)
    layers, _, _ = det._design_to_geometry(design[None])
    layer_z = np.asarray(layers[0], np.float32)
    truth, reco, pa, ei, bz, nf = O._load_reco(n_files)
    # truth layout: [.., vertex(4:7), |p1|(7), p1(8:11), |p2|(11), p2(12:15), ..]
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1)
    # reco layout: [header(0:4), vertex(4:7), p1(7:10), vtx1(10:13), p2(13:16), vtx2(16:19)]
    reco9 = np.concatenate([reco[:, 4:7], reco[:, 7:10], reco[:, 13:16]], axis=1)
    reco_ok = ~np.isnan(reco).any(1)
    rows = np.arange(min(int(n_events), truth.shape[0]))
    ie, bnds = O._input_events(pa, ei, bz, rows)
    traj, ncr, digi, digi_mask = _solver_trajectory(det, ie, bnds, design, layer_z, n_tracks, primaries)
    event, mask = det._trajectory_to_event(traj, ncr, design, hits_xy=True, drift_r=True, tdc=True,
                                           digi=digi, digi_mask=digi_mask)
    fired = (mask > 0)
    match = (np.isfinite(np.asarray(event.tdc)) & fired).sum() / max(fired.sum(), 1)
    n_sec = int((ncr[:, 2:] > 0).any(1).sum()) if n_tracks > 2 else 0
    print(f"  {nf} files: fit {rows.shape[0]} events (material={material}, n_tracks={n_tracks}, "
          f"{n_sec} w/ secondaries); TDC address-match {100 * match:.1f}%")
    return event, mask, true9[rows], reco9[rows], reco_ok[rows]


def _batched_evaluate(tracker, event, mask, chunk):
    """``tracker.evaluate`` over the event axis in chunks (GPU memory), concatenated. Returns
    ``(pred9, doca, nhits, n_stations, chi2)`` -- the per-event predictions + FairShip-cut quantities."""
    B = mask.shape[0]
    parts = []
    for i in range(0, B, chunk):
        parts.append(tracker.evaluate(_slice_event(event, i, i + chunk), mask[i:i + chunk]))
    return tuple(np.concatenate([p[c] for p in parts], 0) for c in range(5))


def _our_accept(d):
    """FairShip-style selection computed from OUR fit: both tracks >= 25 hits, >= 3 stations, chi2/Ndf < 4,
    and 2-track DoCA <= 2 cm."""
    return ((d["nhits"] >= MEAS_CUT).all(1) & (d["n_stations"] >= MIN_STATIONS).all(1)
            & (d["chi2"] < CHI2_CUT).all(1) & (d["doca"] <= DOCA_CUT))


def _fmt(v):
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.3f}"


def _columns(det, store, true9, reco9, reco_ok, kind):
    """One regime's metric dict per column (8 trackers + FairShip). ``kind`` in {phys, r2}. OUR trackers use
    OUR cuts (each method's own accepted subset); the FairShip column uses FAIRSHIP cuts (the events
    FairShip reconstructed)."""
    table = tracking.metrics.metric_table if kind == "phys" else tracking.metrics.r2_table
    cols, results = [], {}
    for name in METHODS:
        d = store[name]
        on = _our_accept(d)
        kw = {"doca": d["doca"]} if kind == "phys" else {}
        m = table(det, d["pred9"], true9, on, **kw)
        m["acceptance %"] = 100.0 * float(on.mean())
        cols.append(CODE[name])
        results[CODE[name]] = m
    m = table(det, reco9, true9, reco_ok)  # FairShip: FairShip cuts (reco'd events); doca n/a (common vertex)
    m["acceptance %"] = 100.0 * float(reco_ok.mean())
    cols.append("FairShip")
    results["FairShip"] = m
    return cols, results


def _render(cols, results, row_names):
    rows = [[r] + [_fmt(results[c].get(r, float("nan"))) for c in cols] for r in row_names]
    return tabulate(rows, headers=["metric", *cols], tablefmt="github", colalign=["left", *(["right"] * len(cols))])


_LEGEND = (
    "FairShip verification: our retina/NLL trackers (4 targets) vs FairShip, on data/mc.\n"
    "Three regimes (hit source varies, all else fixed): our StereoTracking sim material OFF, our sim\n"
    "material ON, and FairShip's OWN digitized daughter hits (real spectrometer field map + geometry\n"
    "calibrated from the MC hit cloud).\n"
    "Columns: R=retina N=NLL | Tu=tubes(V1, wire centre) TD=TDC(V2, raw readout, network proxy)\n"
    "         Dr=drift(V3, smeared r_drift) Hi=hits(V4, exact x,y, sanity floor).  +FairShip = its reco.\n"
    "Cuts: OUR trackers use OUR cuts (>=25 hits/track, >=3 stations, chi2/Ndf<4, DoCA<=2cm, from our\n"
    "      fit); the FairShip column uses FAIRSHIP cuts (events FairShip reconstructed). 'acceptance %'\n"
    "      = fraction passing; all other rows are on each column's accepted subset.\n"
    "Table 1 rows: dp/|p| (median/RMS %); p_x/y/z & vertex_x/y/z (median|err|, RMSE; GeV/cm); doca cm.\n"
    "Table 2 rows: per-component R^2 = 1 - MSE/Var(true) (1=perfect, 0=predict-the-mean, <0=worse).\n")


def _write_report(det0, out, results, regimes, n_events, pending):
    """Render the two master tables over the regimes completed SO FAR (+ a note on what is still running)
    and write ``out`` + the ``results`` .npz. Called after each regime so partial results are on disk."""
    head = _LEGEND + f"Fit on the first {n_events} events. Regimes done: {len(regimes)}/3" + (
        f"; STILL RUNNING: {pending}.\n" if pending else " (COMPLETE).\n")
    sections = []
    for kind, rows_list, title in (
        ("phys", ["acceptance %", *tracking.metrics.ROWS], "MASTER TABLE 1 -- physical units (cm, GeV)"),
        ("r2", ["acceptance %", *tracking.metrics.R2_ROWS], "MASTER TABLE 2 -- R^2 (1=perfect, 0=mean)")):
        blocks = [f"{'=' * 100}\n# {title}\n{'=' * 100}"]
        for _tag, label, store, true9, reco9, reco_ok in regimes:
            cols, results_d = _columns(det0, store, true9, reco9, reco_ok, kind)
            blocks.append(f"## {label}\n" + _render(cols, results_d, rows_list))
        sections.append("\n\n".join(blocks))
    open(out, "w").write(head + "\n" + "\n\n\n".join(sections) + "\n")
    blobs = {}
    for tag, _label, store, true9, reco9, reco_ok in regimes:
        blobs[f"{tag}__true9"], blobs[f"{tag}__reco9"], blobs[f"{tag}__reco_ok"] = true9, reco9, reco_ok
        for name in METHODS:
            for key, val in store[name].items():
                blobs[f"{tag}__{name}__{key}"] = val
    np.savez(results, **blobs)
    print(f"  wrote {out} + {results} ({len(regimes)}/3 regimes)")


def report(n_files=12, n_events=16384, out="report.txt", results="tracking_results.npz", chunk=1024):
    """FairShip-verification master report. Fits all 8 trackers across THREE regimes -- our sim (material
    OFF), our sim (material ON), and FairShip's OWN hits -- writing two master tables to ``out`` (physical
    units + R^2). ``out`` + ``results`` are rewritten AFTER EACH regime so partial results survive a crash.
    Columns = the 8 trackers + FairShip; OUR trackers use OUR cuts, the FairShip column uses FAIRSHIP cuts."""
    det0 = detopt.detector.from_config(yaml.safe_load(open(DETECTOR_CFG)))
    design = np.asarray(det0.flatten_design(yaml.safe_load(open(DESIGN_CFG))), np.float32)
    regimes = []
    todo = ["our sim material OFF", "our sim material ON", "FairShip's own hits"]
    for material in (False, True):
        det = detopt.detector.from_config(yaml.safe_load(open(DETECTOR_CFG)))  # fresh (material is one-way)
        n_tracks, primaries = (PHYS_N_TRACKS, False) if material else (2, True)
        print(f"=== our sim, material {'ON' if material else 'OFF'} ===")
        event, mask, true9, reco9, reco_ok = load_tracking_data(det, design, n_files, n_events, material,
                                                                n_tracks, primaries)
        store = {}
        for name in METHODS:
            tracker = tracking.from_config(yaml.safe_load(open(f"config/tracking/{name}.yaml")), det, design)
            pred9, doca, nhits, n_stations, chi2 = _batched_evaluate(tracker, event, mask, chunk)
            store[name] = dict(pred9=pred9, doca=doca, nhits=nhits, n_stations=n_stations, chi2=chi2)
            print(f"  [{'on' if material else 'off'}] {name:13s} fit done")
        tag = "sim_on" if material else "sim_off"
        label = f"OUR SIM, material = {'ON (secondaries as noise)' if material else 'OFF (clean daughters)'}"
        regimes.append((tag, label, store, true9, reco9, reco_ok))
        todo.pop(0)
        _write_report(det0, out, results, regimes, n_events, ", ".join(todo))  # partial write
    print("=== FairShip's own hits (real field map) ===")
    fs_store, t9, r9, rok = fairship_hits.store(det0, design, n_files, n_events, chunk)
    regimes.append(("fairship", "FAIRSHIP'S OWN HITS (real field map, calibrated geometry)", fs_store, t9, r9, rok))
    _write_report(det0, out, results, regimes, n_events, "")  # final write
    print(f"\ndone: wrote {out}")


def track(config, material=False, n_files=4, n_events=2048, chunk=1024):
    """Fit a single tracker config and print its metrics under both cut regimes (smoke / debug)."""
    det = detopt.detector.from_config(yaml.safe_load(open(DETECTOR_CFG)))
    design = np.asarray(det.flatten_design(yaml.safe_load(open(DESIGN_CFG))), np.float32)
    n_tracks, primaries = (PHYS_N_TRACKS, False) if material else (2, True)
    event, mask, true9, reco9, reco_ok = load_tracking_data(det, design, n_files, n_events, material, n_tracks, primaries)
    cfg = yaml.safe_load(open(config))
    tracker = tracking.from_config(cfg, det, design)
    pred9, doca, nhits, n_stations, chi2 = _batched_evaluate(tracker, event, mask, chunk)
    d = dict(pred9=pred9, doca=doca, nhits=nhits, n_stations=n_stations, chi2=chi2)
    name = list(cfg)[0]
    for cuts, on in (("our", _our_accept(d)), ("fairship", reco_ok)):
        m = tracking.metrics.metric_table(det, pred9, true9, on, doca=doca)
        print(f"{name} material={material} cuts={cuts}: acceptance {100 * on.mean():.0f}%  "
              f"dp/|p| med {m['dp/|p| med%']:.3f}%  RMS {m['dp/|p| RMS%']:.3f}%  doca {m['doca cm med']:.3f}cm")


if __name__ == "__main__":
    import gearup

    gearup.gearup(track=track, report=report)()
