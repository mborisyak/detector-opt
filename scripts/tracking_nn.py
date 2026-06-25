"""Append a trained REGRESSOR (the network) as extra columns to the tracking master tables.

Takes a regressor CHECKPOINT (``scripts/regression.py`` output, e.g. ``data/debug-stereo2``) and an
existing tracking RESULTS archive (``tracking_results.npz`` from ``scripts/tracking.py``) and writes
the two master tables to ``out`` -- the 8 trackers + FairShip columns re-rendered VERBATIM from the
archive (no re-fit), plus two new columns per regime:

    NN     -- the network under OUR cuts (the ``nll-tdc`` tracker's accepted subset; the network has
              no per-track fit of its own, so it borrows that closest-analog selection)
    NN-FS  -- the network under FAIRSHIP cuts (the events FairShip reconstructed)

across all three regimes (our sim material OFF / ON, FairShip's own hits). The network is scored on
the SAME event rows as the archive, fed its NATIVE realistic fired-straw hits: re-simulated with the
C solver for the two sim regimes, and FairShip's own digitized hits for the FairShip regime (a domain
shift -- the network was trained on our sim, so that column is an out-of-distribution estimate).

The regressor ARCHITECTURE is read from the checkpoint itself (so it always matches); the design comes
from ``config/regression.yaml``, and the cut quantities + truth come from the archive. ``n_files`` must match the run that produced the
archive (the default 12 matches ``scripts/tracking.py report``); alignment is asserted on the truth.

    python scripts/tracking_nn.py checkpoint=data/debug-stereo2 results=tracking_results.npz \\
        out=report_nn.txt
"""
import os

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

import detopt
import detopt.tracking as dtracking
import detopt.utils.io
from detopt.detector.straw import StrawEvent
from detopt.utils.config import optimizer as make_optimizer

import tracking as TR  # scripts/tracking.py: solver/render/cut helpers (scripts/ is on sys.path)
import oursim_retina as O  # _load_reco / _input_events / _no_material (sim-regime event loading)
import regression as REG  # _forward_shared (ensemble-averaged forward; checkpoint I/O is in utils.io)

REGIMES = [
    ("sim_off", "OUR SIM, material = OFF (clean daughters)", False),
    ("sim_on", "OUR SIM, material = ON (secondaries as noise)", True),
    ("fairship", "FAIRSHIP'S OWN HITS (real field map, calibrated geometry)", None),
]
NN_REF = "nll-tdc"  # the tracker whose OUR-cut accepted subset the (fit-less) network borrows


# --------------------------------------------------------------------------- #
# Network: load from checkpoint -> physical 9-vec predictor over a StrawEvent.
# --------------------------------------------------------------------------- #
def _load_nn(checkpoint, config, seed):
    """Build the regressor from ``config`` (matching the checkpoint) + restore it. Returns
    ``(detector, theta, predict_norm, step)`` -- ``predict_norm(feats, mask)`` is a jitted deterministic
    (ensemble-averaged) forward to the NORMALIZED target; ``theta`` is the encoded config design."""
    detector = detopt.detector.from_config(config["detector"])
    theta = jnp.asarray(detector.encode_design(config["design"]), jnp.float32)
    rngs = nnx.Rngs(jax.random.PRNGKey(int(seed)))

    manager = detopt.utils.io.get_checkpointer(checkpoint)
    step = manager.latest_step()
    if step is None:
        raise SystemExit(f"no regressor checkpoint to load in {checkpoint}")
    # Architecture from the CHECKPOINT (config files drift); fall back to the config file if absent.
    stored = detopt.utils.io.restore_config(manager)
    regressor_config = stored["regressor"] if stored is not None else config["regressor"]
    model = detopt.nn.from_config(detector, config=regressor_config, rngs=rngs)
    reg_def, params0, state0 = nnx.split(model, nnx.Param, nnx.Variable)
    members = model.ensemble() if hasattr(model, "ensemble") else None
    opt = make_optimizer(config["optimizer"], n_total_steps=1)  # only to shape the restored optimizer state
    params, state, _ = detopt.utils.io.restore_checkpoint(manager, step, regressor=(params0, state0, opt))["regressor"]
    manager.close()

    @jax.jit
    def predict_norm(feats, mask):
        reg = nnx.merge(reg_def, params, state)
        pred = REG._forward_shared(reg, feats, mask, members, deterministic=True)
        return pred if members is None else jnp.mean(pred, axis=0)

    return detector, theta, predict_norm, step


def _nn_pred9(detector, theta, predict_norm, event, mask, chunk):
    """Network predictions over a StrawEvent ``(B, M)`` -> physical ``(B, 9)`` = [vertex(3) cm, p1(3),
    p2(3) GeV], the same layout the trackers + ``true9`` use. ``combine_encoded`` standardises per chunk."""
    B = mask.shape[0]
    preds = []
    for i in range(0, B, chunk):
        ev = TR._slice_event(event, i, i + chunk)
        m = jnp.asarray(mask[i:i + chunk])
        feats = detector.combine_encoded(ev, theta, mask=m)
        preds.append(np.asarray(predict_norm(feats, detector.element_mask(ev, m))))
    pred = detector.denormalize_predictions(np.concatenate(preds, 0))  # DaughterTarget (vertex, p1, p2)
    return np.concatenate([np.asarray(pred.vertex), np.asarray(pred.p1), np.asarray(pred.p2)], axis=-1)


# --------------------------------------------------------------------------- #
# Per-regime network INPUT: the realistic fired-straw StrawEvent on the archive rows.
# --------------------------------------------------------------------------- #
def _sim_digi(config, design_flat, n_files, n_events, material):
    """Re-simulate the first ``n_events`` events at ``design_flat`` and return the fired-straw ``digi``
    (the network's native input) + ``(true9, reco_ok)`` -- aligned to the archive's sim rows. ``material``
    False -> clean daughters (matches the archive's ``sim_off``); True -> secondaries as noise (``sim_on``)."""
    det = detopt.detector.from_config(config["detector"])  # fresh: _no_material is one-way
    if not material:
        O._no_material(det)
    layer_z = np.asarray(det._design_to_geometry(design_flat[None])[0][0], np.float32)
    truth, reco, pa, ei, bz, _nf = O._load_reco(n_files)
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1)
    reco_ok = ~np.isnan(reco).any(1)
    rows = np.arange(min(int(n_events), truth.shape[0]))
    pool, bnds = O._input_events(pa, ei, bz, rows)
    n_tracks, primaries = (TR.PHYS_N_TRACKS, False) if material else (2, True)
    _traj, _ncr, digi, digi_mask = TR._solver_trajectory(det, pool, bnds, design_flat, layer_z, n_tracks, primaries)
    return digi, digi_mask, true9[rows], reco_ok[rows]


def _fairship_digi(detector, n_files, n_events):
    """Pack FairShip's OWN digitized hits into a fired-straw ``StrawEvent`` on the first ``n_events`` rows
    -- ALL valid digis per event (the M earliest-TDC, matching our solver's emission), address mapped to
    our 0-based layout, TDC relative to the event's earliest hit. Returns ``(event, mask, true9, reco_ok)``.
    A domain shift: the network was trained on our sim, not FairShip's readout."""
    from detopt.data.fairship_loader import load_fairship_digi, pack_fairship_events
    data, _nf = load_fairship_digi(n_files)
    event, mask, rows, truth = pack_fairship_events(
        data, n_stations=detector.n_stations, n_views_per_station=detector.n_views_per_station,
        n_layers_per_view=detector.n_layers_per_view, n_straws=detector.n_straws,
        max_hits=detector.max_hits_per_event, n_events=n_events)
    true9 = np.concatenate([truth[:, 4:7], truth[:, 8:11], truth[:, 12:15]], axis=1).astype(np.float32)
    reco_ok = (~np.isnan(np.asarray(data["reco"])).any(1))[rows]
    return event, mask, true9, reco_ok


# --------------------------------------------------------------------------- #
# Table assembly: archive columns + the two network columns.
# --------------------------------------------------------------------------- #
def _nn_columns(det, nn_pred9, true9, store, reco_ok, kind):
    """The two network columns for one regime: ``NN`` under OUR cuts (the ``nll-tdc`` accepted subset,
    since the network has no fit) and ``NN-FS`` under FAIRSHIP cuts (``reco_ok``). ``kind`` in {phys, r2}."""
    table = dtracking.metrics.metric_table if kind == "phys" else dtracking.metrics.r2_table
    res = {}
    for col, on in (("NN", TR._our_accept(store[NN_REF])), ("NN-FS", reco_ok)):
        m = table(det, nn_pred9, true9, on)  # no doca: the network has no fitted tracks -> doca rows n/a
        m["acceptance %"] = 100.0 * float(on.mean())
        res[col] = m
    return ["NN", "NN-FS"], res


def _regime_store(arc, tag):
    """Reconstruct ``(store, true9, reco9, reco_ok)`` for one regime from the archive ``.npz``."""
    keys = ("pred9", "doca", "nhits", "n_stations", "chi2")
    store = {m: {k: arc[f"{tag}__{m}__{k}"] for k in keys} for m in TR.METHODS}
    return store, arc[f"{tag}__true9"], arc[f"{tag}__reco9"], arc[f"{tag}__reco_ok"]


_NN_NOTE = (
    "\nNetwork columns (this report): NN = the trained regressor under OUR cuts (it has no per-track fit,\n"
    f"  so it borrows the {NN_REF} tracker's accepted subset); NN-FS = the same network under FAIRSHIP cuts.\n"
    "  The network is fed its NATIVE realistic fired-straw hits (re-simulated for the sim regimes; FairShip's\n"
    "  own digitized hits for the FairShip regime -- a domain shift, so NN/NN-FS there are an OOD estimate).\n")


def report_nn(checkpoint, results="tracking_results.npz", out="report_nn.txt",
              n_files=12, n_events=None, chunk=1024, seed=0, **config):
    """Write the two master tables to ``out`` with the trackers + FairShip columns taken from the
    ``results`` archive and two new network columns (NN, NN-FS) per regime, scored from ``checkpoint``."""
    detector, theta, predict_norm, step = _load_nn(checkpoint, config, seed)
    design_flat = np.asarray(detector.flatten_design(config["design"]), np.float32)
    arc = dict(np.load(results))
    print(f"loaded regressor step {step} from {checkpoint}; archive {results}")

    # Per regime: the network's input event (its native fired-straw hits) on the archive rows -> pred9.
    nn_pred9 = {}
    for tag, _label, material in REGIMES:
        n_ev = arc[f"{tag}__true9"].shape[0] if n_events is None else int(n_events)
        if tag == "fairship":
            event, mask, true9_chk, _reco_ok = _fairship_digi(detector, n_files, n_ev)
        else:
            event, mask, true9_chk, _reco_ok = _sim_digi(config, design_flat, n_files, n_ev, material)
        a = arc[f"{tag}__true9"]
        if true9_chk.shape != a.shape or not np.allclose(true9_chk, a, atol=1e-3, rtol=0):
            raise SystemExit(f"[{tag}] re-derived truth does not match the archive (shapes {true9_chk.shape} vs "
                             f"{a.shape}); n_files={n_files} probably differs from the archive's run.")
        nn_pred9[tag] = _nn_pred9(detector, theta, predict_norm, event, mask, chunk)
        print(f"  [{tag}] network scored on {mask.shape[0]} events ({int((mask.sum(1) > 0).sum())} with hits)")

    # Render both tables: archive columns (trackers + FairShip) + the NN/NN-FS columns, per regime.
    n_events_done = arc["sim_off__true9"].shape[0]
    head = TR._LEGEND + f"Fit on the first {n_events_done} events. Regimes done: 3/3 (COMPLETE)." + _NN_NOTE
    sections = []
    for kind, rows_list, title in (
        ("phys", ["acceptance %", *dtracking.metrics.ROWS], "MASTER TABLE 1 -- physical units (cm, GeV)"),
        ("r2", ["acceptance %", *dtracking.metrics.R2_ROWS], "MASTER TABLE 2 -- R^2 (1=perfect, 0=mean)")):
        blocks = [f"{'=' * 100}\n# {title}\n{'=' * 100}"]
        for tag, label, _material in REGIMES:
            store, true9, reco9, reco_ok = _regime_store(arc, tag)
            cols, res = TR._columns(detector, store, true9, reco9, reco_ok, kind)
            nn_cols, nn_res = _nn_columns(detector, nn_pred9[tag], true9, store, reco_ok, kind)
            res.update(nn_res)
            trackers = [c for c in cols if c != "FairShip"]  # keep tracker order; FairShip moves next to NN-FS
            ordered = [*trackers, "NN", "FairShip", "NN-FS"]
            blocks.append(f"## {label}\n" + TR._render(ordered, res, rows_list))
        sections.append("\n\n".join(blocks))
    open(out, "w").write(head + "\n" + "\n\n\n".join(sections) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    import gearup

    gearup.gearup(report_nn).with_config("config/regression.yaml")()
