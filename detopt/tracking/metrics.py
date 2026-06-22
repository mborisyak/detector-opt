"""Metric table for a tracking method: ``dp/|p|`` (median%, RMS%) and the per-component momentum/vertex
errors (median |residual| and RMSE), reusing the detector's permutation-matched ``prediction_errors``."""
import numpy as np

_COMPONENTS = ("vertex_x", "vertex_y", "vertex_z", "p_x", "p_y", "p_z")


def metric_table(detector, pred9, true9, on, doca=None):
    """One method's metrics on the selected events ``on`` (bool mask over the full set). ``pred9``/``true9``
    are ``(N, 9)`` = [vertex(3) cm, p1(3) GeV, p2(3) GeV]. ``doca`` (N,) = the two reconstructed tracks'
    closest approach (cm); for FairShip / shared-vertex it is ~0 (tracks meet). Returns named scalar rows."""
    e = detector.prediction_errors(
        np.asarray(detector.normalize_target(pred9[on])), np.asarray(detector.normalize_target(true9[on])))
    out = {}
    for comp in _COMPONENTS:
        r = np.asarray(e[comp][0])  # residuals in physical units (cm / GeV)
        out[f"{comp} med"] = float(np.median(np.abs(r)))
        out[f"{comp} RMSE"] = float(np.sqrt(np.mean(r**2)))
    dp = np.sqrt(sum(np.asarray(e[k][0]) ** 2 for k in ("p_x", "p_y", "p_z")))  # per-daughter |dp|
    pm = np.concatenate([np.linalg.norm(true9[on, 3:6], axis=1), np.linalg.norm(true9[on, 6:9], axis=1)])
    out["dp/|p| med%"] = float(100 * np.median(dp / pm))
    out["dp/|p| RMS%"] = float(100 * np.sqrt(np.mean((dp / pm) ** 2)))
    # DoCA = the two FITTED tracks' closest approach. None -> not measurable (e.g. FairShip's reco gives a
    # single common decay vertex, so its tracks meet by construction) -> NaN (rendered N/A), not a fake 0.
    d = None if doca is None else np.asarray(doca)[on]
    out["doca cm med"] = float("nan") if d is None else float(np.median(d))
    out["doca cm RMSE"] = float("nan") if d is None else float(np.sqrt(np.mean(d**2)))
    return out


# Row order for the report (quantity, then median/RMSE pairs); FairShip shares the same rows.
ROWS = (["dp/|p| med%", "dp/|p| RMS%"]
        + [f"{c} {st}" for c in _COMPONENTS for st in ("med", "RMSE")]
        + ["doca cm med", "doca cm RMSE"])


def r2_table(detector, pred9, true9, on):
    """Per-component R^2 = 1 - MSE / Var(true) on the selected events ``on`` -- a dimensionless score for
    cross-quantity comparison (1 = perfect, 0 = predicts the mean, <0 = worse). Uses the same
    permutation-matched residuals as :func:`metric_table` (momentum pooled over both daughters)."""
    e = detector.prediction_errors(
        np.asarray(detector.normalize_target(pred9[on])), np.asarray(detector.normalize_target(true9[on])))
    t = true9[on]
    true_of = {  # true values per component, aligned with e[comp] (momentum pooled daughter1 then daughter2)
        "vertex_x": t[:, 0], "vertex_y": t[:, 1], "vertex_z": t[:, 2],
        "p_x": np.concatenate([t[:, 3], t[:, 6]]),
        "p_y": np.concatenate([t[:, 4], t[:, 7]]),
        "p_z": np.concatenate([t[:, 5], t[:, 8]]),
    }
    out = {}
    for comp in _COMPONENTS:
        r = np.asarray(e[comp][0])
        var = float(np.var(true_of[comp]))
        out[f"{comp} R2"] = float(1.0 - np.mean(r**2) / var) if var > 0 else float("nan")
    return out


R2_ROWS = [f"{c} R2" for c in _COMPONENTS]
