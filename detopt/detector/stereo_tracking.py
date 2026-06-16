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

from .stereo_straw import StereoStrawDetector

__all__ = ["StereoTracking"]


class StereoTracking(StereoStrawDetector):
    # Pick the daughter ground truth (loader field) as the network target instead of the HNL 6-vec.
    _targets_field = "daughter_targets"

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
            jnp.mean(jnp.square(p2p - p1t) + jnp.square(p1p - p2t), axis=-1)
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

        momenta = (1 - swap)[:, None] * l1_p + swap[:, None] * l2_p

        return {
            "loss": self.loss(predicted, target),
            "vertex_x": vertex[..., 0],
            "vertex_y": vertex[..., 1],
            "vertex_z": vertex[..., 2],
            "p_x": momenta[..., 0],
            "p_y": momenta[..., 1],
            "p_z": momenta[..., 2],
        }
