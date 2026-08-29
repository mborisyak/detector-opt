"""Stereo straw detector with FREE station placement, priced by the intersection of station footprints.

GEOMETRY ENCODING. ``StereoStrawDetector`` maps the scaled cube to station z through SEQUENTIAL COUPLED
windows: station k's window opens one pitch past station k-1, so the stations come out ordered,
non-overlapping and magnet-excluding BY CONSTRUCTION and no point of the cube can violate any of it.
Here each station instead owns a FIXED window that does not depend on its neighbours -- the upstream
stations range from the left detector edge to the magnet face, the downstream ones from the magnet
face to the right detector edge. Within a side the stations are then free to move past one another
and to overlap, and the ordering the sequential map enforced is gone.

THE PRICE OF AN OVERLAP. A pair of stations whose z footprints intersect is not buildable, so the
design carries a COST rather than a bound: :meth:`design_penalty` returns ``overlap_weight`` times the
total pairwise intersection of the station footprints, in cm. Each station occupies ``station_width``
in z, so a pair sharing a z costs a full ``station_width`` and a pair further apart than that costs
nothing. The term is deterministic in the design, so ``scripts/bo.py`` adds it to the trained loss and
leaves the reported error untouched. ``overlap_weight = None`` prices nothing and returns ``None``,
which is a different answer from a price of ``0.0``.

THE MAGNET REMAINS A HARD BOUND, not a priced one: the two sides' windows are disjoint and each keeps
a half station-width clear of the magnet face, so no point of the cube puts a station inside it.

The combine is inherited from ``Stereo4Feature`` unchanged -- ``[TDC, norm z, wire_y_left,
wire_y_right]`` revealed, the 5-feature address withheld.
"""

import jax.numpy as jnp
import numpy as np

from .stereo_tracking import Stereo4Feature

__all__ = ["StereoIntersectionPenalty"]


class StereoIntersectionPenalty(Stereo4Feature):
  """Free per-side station placement + a design price on the station-footprint intersection."""

  def __init__(self, overlap_weight=1.0e-3, **kwargs):
    super().__init__(**kwargs)
    self.overlap_weight = None if overlap_weight is None else float(overlap_weight)

  def _station_lo_hi(self, k, prev_z):
    """Station ``k``'s admissible centre window, INDEPENDENT of its neighbours (``prev_z`` ignored).

    Upstream stations range from the left detector edge to the magnet face, downstream ones from the
    magnet face to the right detector edge, each keeping a half station-width clear of both. Every
    station on a side shares one window, which is what lets stations reorder and overlap."""
    half_width = 0.5 * self.station_width
    min_z, max_z = self.layer_bounds
    if k < self.n_stations_upstream:
      return min_z + half_width, self.z0 - self.magnet_half_cm - half_width
    return self.z0 + self.magnet_half_cm + half_width, max_z - half_width

  def station_intersection(self, design):
    """NOMINAL design -> total pairwise intersection of the station z footprints, in cm.

    Footprint ``k`` spans ``z_k +/- station_width/2``; a pair intersects over
    ``max(0, station_width - |z_i - z_j|)``. What :meth:`design_penalty` prices."""
    stations = self.flatten_design(design)[..., :self.n_stations]
    separation = jnp.abs(stations[..., :, None] - stations[..., None, :])
    pairwise = jnp.maximum(self.station_width - separation, 0.0)
    rows, columns = np.triu_indices(self.n_stations, k=1)
    return jnp.sum(pairwise[..., rows, columns], axis=-1)

  def design_penalty(self, design):
    """``overlap_weight * station_intersection(design)``, or ``None`` when nothing is priced.

    Deterministic in the design, so a caller adds it to the reported loss and leaves the reported
    error alone. ``None`` (no price configured) and ``0.0`` (a price that is zero) differ."""
    if self.overlap_weight is None:
      return None
    return self.overlap_weight * self.station_intersection(design)
