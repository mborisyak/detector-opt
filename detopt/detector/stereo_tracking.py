"""The stereo straw detector with the 4-FEATURE combine -- the default stereo detector.

A combine LEAF: stereo geometry (``StereoStrawDetector``) + the unified daughter objective (the base
``StrawDetector``) + the per-hit 4-feature combine ``[TDC, norm z, wire_y_left, wire_y_right]``, shared
with ``FreeStrawDetector`` through the module function :func:`four_feature_combine`. Its siblings under
``StereoStrawDetector`` (``StereoLayerWise`` / ``Image`` / ``Hits``) only differ in the combine.

``StereoTrackerTruth`` extends it with per-hit TRUTH measurements for tracker experiments.
"""

import numpy as np

from .stereo_straw import StereoStrawDetector
from .straw import StrawEvent, DaughterTarget, four_feature_combine, four_feature_shape

__all__ = ["Stereo4Feature", "DaughterTarget", "StereoTrackerTruth"]


class Stereo4Feature(StereoStrawDetector):
    """Stereo geometry + the 4-feature per-hit combine (the default stereo detector)."""

    def combine_scaled(self, event, design_scaled, mask=None):
        return four_feature_combine(self, event, design_scaled, mask=mask)

    def combined_event_shape(self):
        return four_feature_shape(self)

    def element_mask(self, event, mask):
        return mask  # element == hit (padded hits are gated by the regressor's hit mask)


class StereoTrackerTruth(Stereo4Feature):
    """``Stereo4Feature`` + per-hit TRUTH measurements for tracker experiments: each layer-plane crossing's
    exact ``(x, y)``, the re-derived (smeared) ``drift_r``, and the matched digitised ``tdc``. The base
    ``__call__`` is the clean fired-straw event (inherited, no experiment flags); the TRUTH tracking event
    is the SEPARATE :meth:`tracking_event` (so ``__call__`` keeps the uniform ``(design, event_index)``
    signature -- it is not overridden)."""

    def tracking_event(self, design, event_index, *, n_tracks=2, primaries=True,
                       hits_xy=True, drift_r=True, tdc=True, smear_seed=0):
        """Simulate the events at ``event_index``, RECORDING each track's plane crossings, and return
        ``(ground_truth, tracking_event, mask, target)`` where ``tracking_event`` is a ``StrawEvent``
        carrying the per-hit ``(x, y)`` / ``drift_r`` / matched-``tdc`` truth (``M = n_tracks * m``)."""
        layer_z = np.asarray(self._design_to_geometry(self._resolve_design(design, 1))[0][0], np.float32)
        out = self._simulate(design, event_index, z_planes=layer_z, n_tracks=n_tracks, primaries=primaries)
        event, mask = self._trajectory_to_event(out["traj"], out["n_cross"], design, hits_xy=hits_xy,
                                                 drift_r=drift_r, tdc=tdc, digi=out["X"], digi_mask=out["mask"],
                                                 smear_seed=smear_seed)
        return out["ground_truth"], event, mask, out["target"]

    def _trajectory_to_event(self, traj, n_cross, design, hits_xy=False, drift_r=False, tdc=False,
                             digi=None, digi_mask=None, smear_seed=0):
        """Build a tracking ``StrawEvent`` from the solver's per-track trajectory
        (``traj (B, n_tracks, m, 3)``, ``n_cross (B, n_tracks)``). Each recorded crossing becomes one
        hit: its ``(station, view, layer-in-view)`` address + nearest ``straw`` are quantised from the
        ``(x, y, z)`` crossing (same geometry as :meth:`combine_scaled`), and -- under the flags --
        ``x, y`` (the exact crossing, ``hits_xy``); ``drift_r`` (perpendicular crossing->wire distance,
        re-derived host-side from ``(x, y)`` then SMEARED by ``N(0, sigma_spatial)`` to be FairShip-
        realistic, ``drift_r``); and ``tdc`` (the real digitised TDC, looked up from the ``digi`` packed
        StrawEvent by straw address -- ``NaN`` where the crossing has no matching fired straw). Returns
        ``(StrawEvent, mask)`` with leaves ``(B, M)``, ``M = n_tracks * m``. Uniform design across batch."""
        traj = np.asarray(traj, np.float32)
        n_cross = np.asarray(n_cross)
        B, T, m, _ = traj.shape
        positions, angles, _ = self._design_to_geometry(self._resolve_design(design, 1))
        lz, ang0 = np.asarray(positions[0], np.float32), np.asarray(angles[0], np.float32)  # uniform design
        n_layers = lz.shape[0]
        order = np.argsort(lz)
        # crossing z -> global layer index k (z == lz[k] exactly, copied from z_planes by the solver)
        rank = np.clip(np.searchsorted(lz[order], traj[..., 2]), 0, n_layers - 1)
        k = order[rank]  # (B, T, m) global layer index
        per_station = self.n_views_per_station * self.n_layers_per_view
        station, within = k // per_station, k % per_station
        view, lpv = within // self.n_layers_per_view, within % self.n_layers_per_view
        tan = np.tan(ang0[k])
        cos = 1.0 / np.sqrt(1.0 + tan * tan)
        xx, yy = traj[..., 0], traj[..., 1]
        c = yy - xx * tan  # sheared (wire-frame) coordinate, constant along a wire
        y_stagger = np.where((lpv & 1) == 1, 0.5 * self.layer_y_offset, -0.5 * self.layer_y_offset)
        pitch, height, ns = self.straw_pitch, self.layer_height, self.n_straws
        straw = np.clip(np.round((c + height - y_stagger) / pitch - 0.5), 0, ns - 1)
        wire = (straw + 0.5) * pitch - height + y_stagger  # nearest wire-centre sheared y
        dr = np.abs(c - wire) * cos  # drift radius (perpendicular crossing->wire), re-derived from (x, y)
        if drift_r:  # FairShip-realistic measurement: smear by the single-hit spatial resolution
            rng = np.random.default_rng(smear_seed)
            dr = np.abs(dr + rng.normal(0.0, 0.012, dr.shape))  # 0.012 cm = straw_detector.c STRAW_SIGMA_SPATIAL
        flat = lambda a: a.reshape(B, T * m)
        event = StrawEvent(
            station=flat(station.astype(np.int32)),
            view=flat(view.astype(np.int32)),
            layer=flat(lpv.astype(np.int32)),
            straw=flat(straw.astype(np.int32)),
            tdc=flat(self._match_tdc(k, straw, n_cross, digi, digi_mask)) if tdc
            else flat(np.zeros((B, T, m), np.float32)),  # real TDC by address (V2) else unused
            x=flat(xx) if hits_xy else None,
            y=flat(yy) if hits_xy else None,
            drift_r=flat(dr.astype(np.float32)) if drift_r else None,
        )
        mask = flat((np.arange(m)[None, None, :] < n_cross[:, :, None]).astype(np.int32))
        return event, mask

    def _match_tdc(self, k, straw, n_cross, digi, digi_mask):
        """Real digitised TDC for each trajectory crossing, looked up from the digitised ``digi`` packed
        StrawEvent by global straw key (event, global-layer, straw). ``NaN`` where a crossing has no fired
        straw (capped out / shared). ``k``, ``straw`` are ``(B, T, m)``; returns ``(B, T, m)`` float32."""
        B, T, m = k.shape
        per_station, nlpv, ns = self.n_views_per_station * self.n_layers_per_view, self.n_layers_per_view, self.n_straws
        kpl = self.n_layers * ns  # key span per event
        ck = (np.arange(B)[:, None, None] * kpl + k * ns + straw.astype(np.int64)).reshape(-1)  # crossing keys
        dk_layer = np.asarray(digi.station) * per_station + np.asarray(digi.view) * nlpv + np.asarray(digi.layer)
        dk = (np.arange(B)[:, None] * kpl + dk_layer * ns + np.asarray(digi.straw)).astype(np.int64)  # (B, Md)
        keep = np.asarray(digi_mask) > 0
        dk_f, dt_f = dk[keep], np.asarray(digi.tdc, np.float32)[keep]  # fired digitised straws
        o = np.argsort(dk_f, kind="stable")
        dk_s, dt_s = dk_f[o], dt_f[o]
        pos = np.clip(np.searchsorted(dk_s, ck), 0, max(dk_s.shape[0] - 1, 0))
        hit = (dk_s.shape[0] > 0) & (dk_s[pos] == ck)
        return np.where(hit, dt_s[pos], np.nan).reshape(B, T, m).astype(np.float32)
