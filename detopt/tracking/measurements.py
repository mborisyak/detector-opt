"""The three measurements a tracker can fit, orthogonal to the objective: the activated tube's wire
centre (tubes), the wire centre + drift radius (drift), or the exact (x, y) crossing (hits). Each is a
small class implementing the ``Tracker`` measurement contract -- ``_measurement`` + ``s_lo`` -- and is
mixed with an objective class to form a concrete tracker. The ``StrawEvent`` leaves are host-side numpy
arrays (the detector builds them so); ``_measurement`` returns ``(hit_layer int32, hit_Y, hit_r, valid)``
all as ``(B, M)`` numpy."""
import numpy as np

from ..detector import track_solver


class TubesMeasurement(object):
    """Position-only: the fired tube's wire centre, no drift radius. Resolution = pitch / sqrt(12)."""

    @property
    def s_lo(self):
        return self.sigma_hit

    def _measurement(self, event, mask):
        layer = self.global_layer(event)
        hit_Y = self.wire_sheared_y(event).astype(np.float32)
        return layer, hit_Y, np.zeros_like(hit_Y), (mask > 0).astype(np.float32)


class DriftMeasurement(object):
    """Wire centre + the drift radius (perpendicular crossing->wire distance). Resolution = drift_sigma."""

    @property
    def s_lo(self):
        return self.drift_sigma

    def _measurement(self, event, mask):
        layer = self.global_layer(event)
        hit_Y = self.wire_sheared_y(event).astype(np.float32)
        return layer, hit_Y, event.drift_r, (mask > 0).astype(np.float32)


class HitsMeasurement(object):
    """The exact (x, y) crossing, sheared into the wire frame. Resolution = drift_sigma."""

    @property
    def s_lo(self):
        return self.drift_sigma

    def _measurement(self, event, mask):
        layer = self.global_layer(event)
        hit_Y = (event.y - event.x * self.layer_tan[layer]).astype(np.float32)  # exact sheared crossing
        return layer, hit_Y, np.zeros_like(hit_Y), (mask > 0).astype(np.float32)


class TDCMeasurement(object):
    """V2 -- the raw TDC (ToF + drift + propagation), the readout the network actually sees. The fit's
    measurement is the fired wire's sheared-Y (for seeding) plus the measured TDC; the residual
    (``track_solver.make_tdc_residual``) recovers the drift radius from the TDC and returns a drift-circle
    residual in cm, so V2 shares V3's resolution: sigma_spatial = v_drift * sigma_t."""

    @property
    def s_lo(self):
        return track_solver.STRAW_SIGMA_SPATIAL  # cm (same drift-radius space as V3)

    def _make_residual(self):
        return track_solver.make_tdc_residual(
            self.layer_z, self.layer_tan, self.layer_width, self.z_start, self.field, self.dt, self.n_steps,
            track_solver.STRAW_VDRIFT)

    def _measurement(self, event, mask):
        layer = self.global_layer(event)
        hit_Y = self.wire_sheared_y(event).astype(np.float32)  # fired wire centre (sheared) -- for seeding
        tdc = np.asarray(event.tdc, np.float32)
        valid = (mask > 0) & np.isfinite(tdc)  # only crossings with a matched digitised straw carry a TDC
        return layer, hit_Y, np.where(valid, tdc, 0.0).astype(np.float32), valid.astype(np.float32)
