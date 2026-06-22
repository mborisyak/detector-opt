"""Stereo straw detector whose regression target is the two DAUGHTER particles' kinematics
instead of the HNL's.

Same stereo design (``[station_z(n), stereo_angle]``, field fixed at ``max_B``) and same HNL
conditioning as :class:`StereoStrawDetector`; it only PICKS a different sampled ground truth as
the network target -- the two daughters' ``[decay_vertex(3), product1_p(3), product2_p(3)]``
(9-vec, the daughters share the decay vertex) -- by setting ``_targets_field``. The base
``StrawDetector`` owns the normalization constants, target shape and metric labels for both
fields; only the LOSS is daughter-specific and lives here.

The two daughters are PERMUTATION-INVARIANT (their labelling p1/p2 is arbitrary), so the loss is
the minimum over the two assignments of predicted->true momenta (the shared vertex is not
permuted). The base normalizes both momentum slots with the SAME constants, so the swap is exact.
"""

from typing import NamedTuple

import jax
import numpy as np

from .stereo_straw import StereoStrawDetector

__all__ = ["StereoTracking", "DaughterTarget"]


class DaughterTarget(NamedTuple):
    """Daughter-tracking target: shared decay vertex (cm) + the two daughters' momenta (GeV)."""

    vertex: jax.Array  # (..., 3)
    p1: jax.Array  # (..., 3)
    p2: jax.Array  # (..., 3)


class StereoTracking(StereoStrawDetector):
    # Pick the daughter ground truth (loader field) as the network target instead of the HNL 6-vec.
    _targets_field = "daughter_targets"

    def target_spec(self):
        # Daughter 9-vec: shared decay vertex + the two daughters' momenta.
        f = lambda n: jax.ShapeDtypeStruct((n,), np.float32)
        return DaughterTarget(vertex=f(3), p1=f(3), p2=f(3))

    def _pack_target(self, targets):
        """Pack the daughter target array ``(n, 9)`` [vertex, p1, p2] into a ``DaughterTarget``."""
        t = np.asarray(targets)
        return DaughterTarget(vertex=t[..., :3], p1=t[..., 3:6], p2=t[..., 6:9])

    def metric_labels(self):
        """Keys of the metric() dict, in display order (p_{x,y,z} is shared over the daughters)."""
        return ("loss", "vertex_x", "vertex_y", "vertex_z", "p_x", "p_y", "p_z")

    def loss(self, predicted, target):
        """Per-sample MSE on the normalized 9-vec, PERMUTATION-INVARIANT over the two daughters:
        the shared vertex is matched directly; the momenta are matched by the cheaper of the two
        assignments (p1<->p1,p2<->p2) vs (p1<->p2,p2<->p1). Mean over all 9 components, so on the
        same scale as the 6-vec base MSE. Broadcasts over any leading axes (e.g. ``(members, B)``)."""
        import jax.numpy as jnp

        vp, vt = predicted[..., :3], target[..., :3]
        p1p, p2p = predicted[..., 3:6], predicted[..., 6:9]
        p1t, p2t = target[..., 3:6], target[..., 6:9]

        loss_vertex = 0.5 * jnp.mean(jnp.square(vp - vt), axis=-1)
        loss_momenta = 0.25 * jnp.minimum(
            jnp.mean(jnp.square(p1p - p1t) + jnp.square(p2p - p2t), axis=-1),
            jnp.mean(jnp.square(p2p - p1t) + jnp.square(p1p - p2t), axis=-1),
        )

        return loss_vertex + loss_momenta

    def metric(self, predicted, target):
        """Per-sample diagnostics dict on the normalized target: overall ``loss`` plus per-component
        squared error -- ``vertex_{x,y,z}`` and the SHARED daughter momentum ``p_{x,y,z}``, which is
        ``0.5 * (matched p1 + matched p2)`` under the best (loss-minimizing) daughter assignment."""
        import jax.numpy as jnp

        vp, vt = predicted[..., :3], target[..., :3]
        p1p, p2p = predicted[..., 3:6], predicted[..., 6:9]
        p1t, p2t = target[..., 3:6], target[..., 6:9]

        vertex = jnp.square(vp - vt)
        l1_p = jnp.square(p1p - p1t) + jnp.square(p2p - p2t)
        l2_p = jnp.square(p2p - p1t) + jnp.square(p1p - p2t)
        swap = jnp.sum(l1_p, axis=-1) > jnp.sum(l2_p, axis=-1)

        momenta = (1 - swap)[..., None] * l1_p + swap[..., None] * l2_p

        return {
            "loss": self.loss(predicted, target),
            "vertex_x": vertex[..., 0],
            "vertex_y": vertex[..., 1],
            "vertex_z": vertex[..., 2],
            "p_x": momenta[..., 0],
            "p_y": momenta[..., 1],
            "p_z": momenta[..., 2],
        }

    def prediction_errors(self, predicted_norm, target_norm):
        """Signed real-unit residuals ``predicted - true`` per quantity, for error histograms.
        The two daughters are PERMUTATION-INVARIANT, so each sample's momenta are matched by the
        loss-minimizing assignment (as in :meth:`metric`) before differencing, then both matched
        daughters are POOLED into ``p_{x,y,z}``. Inputs are NORMALIZED ``(N, 9)``; returns
        ``{label: (errors, unit)}`` -- vertex in cm (N), momentum in GeV (2N, both daughters)."""
        import numpy as np

        pred = self.denormalize_predictions(predicted_norm)  # DaughterTarget
        true = self.denormalize_predictions(target_norm)
        vertex = np.asarray(pred.vertex) - np.asarray(true.vertex)
        p1p, p2p = np.asarray(pred.p1), np.asarray(pred.p2)
        p1t, p2t = np.asarray(true.p1), np.asarray(true.p2)
        l_direct = np.sum((p1p - p1t) ** 2 + (p2p - p2t) ** 2, axis=-1)
        l_swap = np.sum((p2p - p1t) ** 2 + (p1p - p2t) ** 2, axis=-1)
        swap = (l_swap < l_direct)[:, None]
        err1 = np.where(swap, p2p - p1t, p1p - p1t)  # residual against true daughter 1
        err2 = np.where(swap, p1p - p2t, p2p - p2t)  # residual against true daughter 2
        mom = np.concatenate([err1, err2], axis=0)  # pooled over both daughters -> (2N, 3)
        return {
            "vertex_x": (vertex[:, 0], "cm"),
            "vertex_y": (vertex[:, 1], "cm"),
            "vertex_z": (vertex[:, 2], "cm"),
            "p_x": (mom[:, 0], "GeV"),
            "p_y": (mom[:, 1], "GeV"),
            "p_z": (mom[:, 2], "GeV"),
        }

    def metric_real_rmse(self, metric_means, metric_errors=None):
        """Sample-averaged normalized per-component metric (from :meth:`metric`) -> real-unit RMSE.

        ``vertex_{x,y,z}`` are single normalized squared errors, so RMSE = ``sqrt(mse) * decay_sigma``
        in cm. ``p_{x,y,z}`` are the SUMMED squared error over the two daughters, so the per-daughter
        RMSE is ``sqrt(mse / 2) * daughter_momentum_sigma`` in GeV. ``loss`` (a mixed normalized
        quantity) has no single physical unit and is omitted.

        If ``metric_errors`` (the standard error on each mean MSE) is given, its 1-sigma uncertainty
        is propagated to the RMSE: with ``RMSE = sqrt(mse/n)*sigma``, ``d(RMSE) = sigma * sem /
        (2*sqrt(n*mse))`` -- reported as ``error`` (cm / GeV)."""
        import numpy as np

        ds = np.asarray(self.decay_sigma, np.float64)
        dps = np.asarray(self.daughter_momentum_sigma, np.float64)
        spec = {  # key: (sigma, n_daughters_summed, unit)
            "vertex_x": (ds[0], 1, "cm"),
            "vertex_y": (ds[1], 1, "cm"),
            "vertex_z": (ds[2], 1, "cm"),
            "p_x": (dps[0], 2, "GeV"),
            "p_y": (dps[1], 2, "GeV"),
            "p_z": (dps[2], 2, "GeV"),
        }
        out = {}
        for k, (sigma, n, unit) in spec.items():
            if k not in metric_means:
                continue
            mse = float(metric_means[k])
            entry = {"rmse": float(np.sqrt(mse / n) * sigma), "unit": unit}
            if metric_errors is not None and k in metric_errors:
                sem = float(metric_errors[k])
                entry["error"] = float(sigma * sem / (2.0 * np.sqrt(n * mse))) if mse > 0 else float("inf")
            out[k] = entry
        return out
