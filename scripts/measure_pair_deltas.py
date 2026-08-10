"""Measure HNL-daughter pair-wise feature deltas (ISOLATED) for the pair set regressor.

Hits left by the two HNL daughters are nearly identical -- a clean track fires one straw per layer
and neighbouring straws differ only slightly in drift TDC and wire-y. The pair set regressor
amplifies those tiny differences with ``tanh(coeff_f * (feat_i - feat_j))``, so each feature needs
a coefficient ~ ``1 / (typical small daughter Δfeature)`` to map a typical difference into the
visible (non-vanishing, non-saturated) part of the ``tanh``.

This script simulates events at the config design, keeps ONLY daughter hits (primary process,
``process_id == 0``; see the primary-hits note), and reports per :meth:`combine` feature the
typical pair-wise difference between daughter hits in the same event -- both the nearest-neighbour
delta (the smallest resolvable scale to amplify) and the median over all daughter pairs. It then
prints a ready-to-paste ``coefficients:`` line (reciprocal of the nearest-neighbour delta) for the
network config in ``config/regression.yaml``.

    python scripts/measure_pair_deltas.py seed=0 n_events=2048

This touches only the detector's existing public interface -- it does not modify any detector.
"""

import numpy as np
import jax.numpy as jnp

import detopt
from detopt.utils.events import shuffled_event_index

# combine() -> [TDC, norm(layer z), wire_y_left, wire_y_right]
FEATURE_LABELS = ("tdc", "norm_z", "wire_y_left", "wire_y_right")


def measure(seed=0, n_events=2048, **config):
    detector = detopt.detector.from_config(config["detector"])
    F = detector.combined_feature_dim
    M = detector.max_hits_per_event

    theta = jnp.asarray(detector.to_scaled(config["design"]), jnp.float32)  # the design we train at
    phys = np.asarray(detector.flatten_design(detector.to_nominal(theta)), np.float32)
    design = np.broadcast_to(phys[None, :], (n_events, phys.shape[0]))  # one row per event

    event_index = shuffled_event_index(detector.size(), int(n_events), int(seed))
    pool, boundaries, _t, _c = detector._events_at(event_index)
    layers, angles, Bs = detector._design_to_geometry(design)
    process_ids = np.zeros((n_events, M), dtype=np.int32)
    hits_idx, tdc, _ = detector._run_solver(pool, boundaries, layers, angles, Bs,
                                            detector._seeds(event_index), process_ids=process_ids)
    mask = (tdc >= 0).astype(np.int32)

    theta_b = jnp.broadcast_to(theta[None, :], (n_events, detector.design_dim()))
    event = detector._pack_event(hits_idx, tdc)  # -> StrawEvent
    feats = np.asarray(detector.combine_scaled(event, theta_b))  # (n, M, F)

    pairwise = [[] for _ in range(F)]  # all same-event daughter-pair |Δ| per feature
    daughters_per_event = []
    for e in range(n_events):
        sel = (process_ids[e] == 0) & (mask[e] > 0)  # daughter (primary) hits only
        D = feats[e][sel]  # (k, F)
        daughters_per_event.append(len(D))
        if len(D) < 2:
            continue
        iu = np.triu_indices(len(D), k=1)  # unordered pairs, no self
        diff = np.abs(D[iu[0]] - D[iu[1]])  # (pairs, F)
        for f in range(F):
            pairwise[f].append(diff[:, f])

    pairwise = [np.concatenate(v) if v else np.array([np.nan]) for v in pairwise]

    print(f"design: {detector.to_nominal(theta)}")
    print(
        f"events={n_events}  daughter hits/event: mean={np.mean(daughters_per_event):.1f} "
        f"median={int(np.median(daughters_per_event))}\n"
    )
    # The daughter-pair Δ is the small scale to amplify; coeff = 1/median(Δ) maps a typical
    # difference to tanh(1) ~ 0.76 (visible, not yet saturated). A near-zero Δ (e.g. norm_z, equal
    # for same-layer hits) -> coeff 0 (comparison off for that feature).
    print(f"{'feature':>14s} {'Δ median':>12s} {'Δ mean':>12s} {'-> coeff = 1/median':>20s}")
    coeffs = []
    for f in range(F):
        med = float(np.median(pairwise[f]))
        coeff = 1.0 / med if med > 1e-9 else 0.0
        coeffs.append(coeff)
        print(f"{FEATURE_LABELS[f]:>14s} {med:>12.5g} {float(np.mean(pairwise[f])):>12.5g} {coeff:>20.5g}")

    line = "[ " + ", ".join(f"{c:.5g}" for c in coeffs) + " ]"
    print(f"\npaste into config/regression.yaml (regressor.pair-set-regressor.coefficients):\n  coefficients: {line}")
    return coeffs


if __name__ == "__main__":
    import gearup

    gearup.gearup(measure).with_config("config/regression.yaml")()
