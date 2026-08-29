"""Stereo straw detector whose ONLY design dof is the shared stereo view angle.

THE STATIONS ARE FIXED GEOMETRY HERE, NOT A DESIGN. ``fixed_stations`` is a construction parameter
like ``z0`` or ``magnet_half_cm``: it is required from the config, it is never proposed, and the
detector stores no design of any kind. The intended value is the best station placement found by the
1M ``ship-addr-prec1e2`` corpus, which is a CORNER of that campaign's box -- the stations were pushed
as far apart as their windows allowed on both sides of the magnet -- so a run here explores the angle
at a placement the search had already pinned to its bound.

THE ANGLE RANGES OVER A SIGNED INTERVAL. ``stereo_bound`` is ``(-angle_max, +angle_max)`` rather than
the ``(0, angle_max)`` the station-plus-angle designs use, so the search covers both stereo
handednesses. The per-station view pattern stays ``[0, +a, -a, 0]``, so a negative ``a`` swaps which
of the two stereo views leans which way -- a reflection of the layout, not a new one, and the pair of
signed halves is therefore a symmetry the GP sees as two arms of one function.

The combine is inherited from ``Stereo4Feature`` unchanged -- ``[TDC, norm z, wire_y_left,
wire_y_right]`` revealed, the 5-feature address withheld.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .stereo_straw import StereoDesign
from .stereo_tracking import Stereo4Feature
from .straw import scale, unscale

__all__ = ["StereoAngleOnly", "AngleDesign"]


class AngleDesign(NamedTuple):
  """One-dof design: the shared stereo view angle in radians."""

  view_angle: jax.Array


class StereoAngleOnly(Stereo4Feature):
  """Fixed station placement + the stereo angle as the single design dof."""

  def __init__(self, fixed_stations=None, **kwargs):
    super().__init__(**kwargs)
    if fixed_stations is None:
      raise ValueError("fixed_stations is required: the placement is fixed geometry and must come from the config")
    stations = np.asarray(fixed_stations, np.float32)
    if stations.shape != (self.n_stations, ):
      raise ValueError(f"fixed_stations must hold {self.n_stations} station z values, got shape {stations.shape}")
    self.fixed_stations = stations

  def design_shape(self):
    return (1, )

  def design_spec(self):
    return AngleDesign(view_angle=jax.ShapeDtypeStruct((1, ), np.float32))

  def design_bounds(self):
    return {"view_angle": (self.stereo_bound[0], self.stereo_bound[1])}

  def _as_stereo_design(self, design, xp):
    """Angle design -- ``AngleDesign``, ``StereoDesign`` or a flat ``[view_angle]`` -- to a BATCHED
    ``StereoDesign`` carrying the fixed stations, so ``_expand`` and the combine read it by name."""
    if isinstance(design, StereoDesign):
      return design
    angle = design.view_angle if isinstance(design, AngleDesign) else design
    d = xp.asarray(angle)
    d = d if d.ndim == 2 else xp.reshape(d, (1, -1))
    stations = xp.broadcast_to(xp.asarray(self.fixed_stations, dtype=d.dtype), (d.shape[0], self.n_stations))
    return StereoDesign(stations=stations, view_angle=d[:, :1])

  def _to_scaled_flat(self, design):
    d = jnp.asarray(design, jnp.float32)
    return scale(d[..., :1], self.stereo_bound)

  def _to_nominal_flat(self, design_scaled):
    u = jnp.asarray(design_scaled, jnp.float32)
    return unscale(u[..., :1], self.stereo_bound)
