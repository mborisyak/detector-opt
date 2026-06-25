"""Shared host-side loader/packer for FairShip's OWN digitized straw hits (``Ship2NumPy.py`` npz).

Used by the ``SHiPRelayDetector`` (replay FairShip hits, no simulation) and by the evaluation
scripts (``scripts/fairship_hits.py``, ``scripts/tracking_nn.py``). ``load_fairship_digi`` concatenates
the per-file arrays; ``pack_fairship_events`` packs the valid digis of each event into a fired-straw
``StrawEvent`` (0-based address, the ``max_hits`` earliest-TDC hits, TDC relative to the event's
earliest hit) -- the exact packing the network's combine consumes.
"""
import glob

import numpy as np

from ..detector.straw import StrawEvent

__all__ = ["load_fairship_digi", "pack_fairship_events"]

# Columns the relay/packer needs; callers wanting the MC truth hit cloud (fairship_hits) pass their own.
_DEFAULT_COLUMNS = ("truth", "reco", "digi_straw", "digi_invalid", "digi_event_index", "digi_tdc")


def load_fairship_digi(n_files=None, data_glob="data/mc/*.npz", columns=_DEFAULT_COLUMNS):
    """Concatenate the npz files (those carrying a ``reco`` field) under ``data_glob``. ``n_files=None``
    loads ALL of them; otherwise the first ``n_files``.

    Returns ``(data, n_files_used)`` where ``data`` maps each requested column to a concatenated array.
    ``digi_event_index`` is offset per file so it indexes the concatenated ``truth`` rows globally."""
    files = [f for f in sorted(glob.glob(data_glob)) if "reco" in np.load(f, allow_pickle=True).files]
    if n_files is not None:
        files = files[: int(n_files)]
    cols = {k: [] for k in columns}
    offset = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        for k in cols:
            cols[k].append(np.asarray(d[k]))
        if "digi_event_index" in cols:
            cols["digi_event_index"][-1] = cols["digi_event_index"][-1] + offset
        offset += d["truth"].shape[0]
    out = {k: np.concatenate(v) for k, v in cols.items()}
    return out, len(files)


def pack_fairship_events(data, *, n_stations, n_views_per_station, n_layers_per_view, n_straws,
                         max_hits, n_events=None, tdc_clip=1.0e4):
    """Pack FairShip's valid digis into a fired-straw ``StrawEvent`` over the first ``n_events`` events.

    Per event, the ``max_hits`` earliest-TDC valid digis (matching our solver's emission) are kept;
    the FairShip address (station 1..n / straw 1..n_straws, 1-based; view/layer 0-based) is mapped to
    our 0-based layout; the absolute FairShip TDC is made relative to the event's earliest hit and
    clipped to ``tdc_clip`` (FairShip has 1e16 outliers). Returns ``(event, mask, rows, truth)`` with
    leaves ``(len(rows), max_hits)``; ``rows`` are the kept ``truth`` row indices."""
    ds = np.asarray(data["digi_straw"])
    tdc = np.asarray(data["digi_tdc"], np.float64)
    deidx = np.asarray(data["digi_event_index"])
    valid = ~np.asarray(data["digi_invalid"])
    truth = np.asarray(data["truth"])
    n_total = truth.shape[0]
    rows = np.arange(n_total if n_events is None else min(int(n_events), n_total))
    M = int(max_hits)

    st = np.clip(ds[:, 0].astype(np.int64) - 1, 0, n_stations - 1)
    vw = np.clip(ds[:, 1].astype(np.int64), 0, n_views_per_station - 1)
    ly = np.clip(ds[:, 2].astype(np.int64), 0, n_layers_per_view - 1)
    sw = np.clip(ds[:, 3].astype(np.int64) - 1, 0, n_straws - 1)

    o = np.argsort(deidx[valid], kind="stable")  # group valid digis by event
    e_s = deidx[valid][o]
    st_s, vw_s, ly_s, sw_s, td_s = st[valid][o], vw[valid][o], ly[valid][o], sw[valid][o], tdc[valid][o]

    cols = {k: np.zeros((len(rows), M), dt) for k, dt in
            (("station", np.int32), ("view", np.int32), ("layer", np.int32), ("straw", np.int32), ("tdc", np.float32))}
    mask = np.zeros((len(rows), M), np.int32)
    for r, e in enumerate(rows):
        lo, hi = np.searchsorted(e_s, e, "left"), np.searchsorted(e_s, e, "right")
        if hi <= lo:
            continue
        sel = lo + np.argsort(td_s[lo:hi])[:M]  # the M earliest-TDC hits
        n = sel.shape[0]
        cols["station"][r, :n], cols["view"][r, :n] = st_s[sel], vw_s[sel]
        cols["layer"][r, :n], cols["straw"][r, :n] = ly_s[sel], sw_s[sel]
        t = td_s[sel] - td_s[sel].min()  # FairShip TDC is absolute -> relative to the event's earliest hit
        cols["tdc"][r, :n] = np.clip(t, 0.0, tdc_clip).astype(np.float32)
        mask[r, :n] = 1
    event = StrawEvent(station=cols["station"], view=cols["view"], layer=cols["layer"],
                       straw=cols["straw"], tdc=cols["tdc"])
    return event, mask, rows, truth[rows]
