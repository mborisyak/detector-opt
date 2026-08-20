"""Digit classification from a 10x10 SAMPLING GRID whose point DENSITY is the design.

The frame is treated as a continuous field -- a bicubic (Keys / Catmull-Rom cubic convolution)
interpolant through the pixel centres -- and read out at ``n_grid^2`` points. The budget of points is
FIXED by ``n_grid``, so the design cannot buy resolution; it can only decide where the points go.

* **design** -- ``(sigma_x, sigma_y)``, two coordinates on ``[sigma_min, sigma_max]``, in units of
  the image's HALF-WIDTH. The probe DENSITY along each axis is ``N(0, sigma^2)``: with
  ``t = linspace(0, 1, n_grid)`` (endpoints included) the positions are the quantiles of that normal
  TRUNCATED to the frame, ``x_i = sigma * Phi^-1(lo + t_i * (hi - lo))`` with ``lo = Phi(-1/sigma)``
  and ``hi = Phi(1/sigma)``. Equal probability mass between consecutive probes means the spacing goes
  as ``1 / pdf``, which IS the requirement. ``x_0 = -1`` and ``x_{n-1} = +1`` exactly, at every
  design, so the outermost probes sit on the image's corners and none is ever spent outside it, and
  ``sigma`` moves only what is between them: as ``sigma -> 0`` the interior collapses onto the
  CENTRE, so the network sees a magnified crop of the middle and the digit's extremities are never
  looked at; as ``sigma -> infinity`` the Gaussian is flat across the frame and the grid tends to
  UNIFORM. NOTHING EVER CLUSTERS AT THE EDGE -- a Gaussian density cannot do it, and an
  implementation that does is wrong. SCALED is log-affine over the bounds, since ``sigma`` is a
  scale.
  ⚠️ THE TRUNCATION IS WHAT MAKES ``sigma`` BITE. Warping a FIXED set of interior quantiles and then
  renormalising by the extreme points looks equivalent and is not: as ``sigma -> 0`` that family
  divides ``sigma`` straight back out, every small design gives an identical grid, and the detector
  is design-blind over half its box.
* **event** -- one whole frame, ``(n_rows, n_columns)`` ``uint8``. It does NOT depend on the design:
  the sampling happens in :meth:`combine_scaled`, so one pool of frames serves every design and no
  design can move its own label.
* **combine** -- ``(n_grid, n_grid, 3)`` float32: the sampled field on roughly ``[0, 1]`` beside two
  constant planes holding the SCALED ``sigma_x`` and ``sigma_y``. The design planes are what let the
  network read the grid, whose spacing is not uniform and which only ``sigma`` names. Dense --
  every element is real and nothing is masked. Cubic convolution RINGS, so channel 0 can leave
  ``[0, 1]`` by a little near a sharp stroke; that overshoot is the interpolant and is not clipped.
* **target** -- the digit, one-hot over ``N_CLASSES``.
* **loss** -- softmax cross-entropy in nats, undivided; a uniform prediction scores ``ln(n_classes)``.
* **price** -- NONE. :meth:`design_penalty` returns ``None``, the absence of a price rather than a
  price of ``0.0``. The probe budget is fixed at ``n_grid^2`` and the field of view is fixed at the
  whole frame, so the trade-off is already intrinsic: a probe moved into the middle is a probe taken
  from the outside. Nothing needs to be charged for.
  ⚠️ CAPTURED INK IS NOT THE OBJECTIVE. Concentrating the probes on the dense central stroke
  MAXIMISES the intensity read out, and at the smallest ``sigma`` the sampled digit is unreadable
  while scoring the most ink. Only the classification loss locates the optimum.

OUTSIDE THE FRAME IS ZERO. The interpolant's 4x4 stencil reaches past the pixel-centre lattice near
the border, and a tap outside it contributes 0 -- the black background continued. This is exact
rather than incidental: the separable weight matrix ``W(position - pixel_index)`` simply has no
column for a tap that does not exist, so no index is clamped, wrapped or left to produce a NaN, and
the field decays smoothly to 0 across the outermost half-pixel. The corner probes live in exactly
that half-pixel -- ``x = -1`` is pixel coordinate ``-0.5`` -- so they read HALF the boundary value per
axis and a QUARTER at a corner. The frame's border is background on these digits, so that costs no
information; it is stated because it now holds at every design rather than at an extreme one.

COORDINATE CONVENTION: the frame is the SYMMETRIC box ``[-1, 1]`` on each axis, centred on the image
centre, ``x`` along the COLUMN axis and ``y`` along the ROW axis; ``sigma`` is therefore in units of
the half-width. Pixel ``k`` of ``n`` covers ``[-1 + 2k/n, -1 + 2(k + 1)/n)``, so its centre is at
``-1 + (2k + 1)/n`` and a position maps to the continuous pixel coordinate
``p = (x + 1) * n / 2 - 0.5``.

⚠️ EMNIST frames are stored column-major and :func:`read_idx` transposes them; this detector must
NOT transpose again, the operation being its own inverse. Frames arrive through ``read_arrow``
(``data_path`` alone) or ``read_idx`` (plus ``labels_path``).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import Detector
from .mnist import N_CLASSES, MNISTEvent, MNISTGroundTruth, MNISTTarget, read_arrow, read_idx
from ..utils import tensor

__all__ = ['PixelDensityDetector', 'PixelDensityDesign', 'KEYS_A', 'HALF_WIDTH', 'keys_kernel']

# Keys' cubic convolution parameter. -1/2 is the Catmull-Rom member of the family, the one whose
# interpolant reproduces every polynomial of degree <= 2 per axis exactly; it is a property of the
# kernel, not a tuning knob, so it is fixed here rather than exposed as a constructor argument.
KEYS_A = -0.5

# The frame is the symmetric box [-HALF_WIDTH, HALF_WIDTH] on each axis, so `sigma` is in units of
# the image's HALF-WIDTH. TAIL_FLOOR is a floor on the inverse CDF's argument: it only guards the
# arithmetic (the probes it would affect are the extremes, which are pinned to the bounds anyway),
# and it is what keeps an arbitrarily small `sigma` from reaching Phi^-1(0) = -inf.
HALF_WIDTH = 1.0
TAIL_FLOOR = 1.0e-7


def keys_kernel(distance):
  """Keys' cubic convolution kernel at ``KEYS_A``, evaluated on a signed distance in PIXELS.

  Support ``|s| < 2``, so four taps per axis; zero outside, which is what makes the separable weight
  matrix drop out-of-frame taps instead of clamping them."""
  s = jnp.abs(jnp.asarray(distance, jnp.float32))
  near = ((KEYS_A + 2.0) * s - (KEYS_A + 3.0)) * s * s + 1.0
  far = ((KEYS_A * s - 5.0 * KEYS_A) * s + 8.0 * KEYS_A) * s - 4.0 * KEYS_A
  return jnp.where(s < 1.0, near, jnp.where(s < 2.0, far, 0.0))


class PixelDensityDesign(NamedTuple):
  """The two concentrations, NOMINAL: ``sigma_x`` along the column axis and ``sigma_y`` along the row
  axis, each ``(1,)`` on ``[sigma_min, sigma_max]``. The flat design is ``(sigma_x, sigma_y)``.

  Two fields rather than one ``(2,)`` field because the axes are NOT exchangeable: a digit is not
  invariant under transposing the frame, so no kernel should be told they are."""
  sigma_x: jax.Array
  sigma_y: jax.Array


class PixelDensityDetector(Detector):
  """A bicubic interpolant through a digit, read out on an ``n_grid x n_grid`` grid whose point
  density is the design (see the module docstring).

  Every constant is a constructor argument: the frame source, how many frames to read, the number of
  grid points per axis, and the ``sigma`` range.
  """

  def __init__(
    self, *, data_path: str, labels_path: str | None = None, n_events: int | None = None, n_grid: int = 10,
    sigma_min: float = 0.2, sigma_max: float = 5.0
  ):
    self.data_path = str(data_path)
    self.labels_path = None if labels_path is None else str(labels_path)
    self.n_events = None if n_events is None else int(n_events)
    self.n_grid = int(n_grid)
    self.sigma_min = float(sigma_min)
    self.sigma_max = float(sigma_max)
    self.n_classes = N_CLASSES
    if self.n_events is not None and self.n_events < 1:
      raise ValueError(f'n_events must be None or a positive int, got {n_events}')
    if self.n_grid < 2:
      raise ValueError(f'n_grid must be at least 2 to make a grid, got {n_grid}')
    if not 0.0 < self.sigma_min < self.sigma_max:
      raise ValueError(f'the sigma range must satisfy 0 < sigma_min < sigma_max, got ({sigma_min}, {sigma_max})')

    if self.labels_path is None:
      images, labels = read_arrow(self.data_path, self.n_events)
    else:
      images, labels = read_idx(self.data_path, self.labels_path, self.n_events)
    if images.ndim != 3:
      raise ValueError(f'{self.data_path} decoded to {images.shape}; expected (N, rows, columns) single-channel images')
    if labels.min() < 0 or labels.max() >= self.n_classes:
      raise ValueError(f'{self.data_path} holds labels in [{labels.min()}, {labels.max()}], outside [0, {self.n_classes})')
    self.images = images
    self.labels = labels
    self.n_rows = int(images.shape[1])
    self.n_columns = int(images.shape[2])
    self.one_hot = np.eye(self.n_classes, dtype=np.float32)[labels]

    # The probability LEVELS are the constants of n_grid alone and are built once, here; the
    # truncation bounds move with the design, so the inverse CDF itself cannot be precomputed.
    # `edge_sign` marks the two extreme probes, which ARE the truncation bounds and are pinned.
    levels = np.linspace(0.0, 1.0, self.n_grid)
    self.levels = jnp.asarray(levels, jnp.float32)
    edge_sign = np.zeros(self.n_grid, np.float32)
    edge_sign[0], edge_sign[-1] = -1.0, 1.0
    self.edge_sign = jnp.asarray(edge_sign, jnp.float32)
    self.row_index = jnp.arange(self.n_rows, dtype=jnp.float32)
    self.column_index = jnp.arange(self.n_columns, dtype=jnp.float32)
    self.log_sigma_min = float(np.log(self.sigma_min))
    self.log_sigma_span = float(np.log(self.sigma_max) - np.log(self.sigma_min))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return MNISTEvent(image=jax.ShapeDtypeStruct((self.n_rows, self.n_columns), np.uint8))

  def target_spec(self):
    return MNISTTarget(digit=jax.ShapeDtypeStruct((self.n_classes, ), np.float32))

  def ground_truth_spec(self):
    return MNISTGroundTruth(digit=jax.ShapeDtypeStruct((self.n_classes, ), np.float32))

  def design_shape(self):
    return (2, )

  def design_spec(self):
    scalar = jax.ShapeDtypeStruct((1, ), np.float32)
    return PixelDensityDesign(sigma_x=scalar, sigma_y=scalar)

  def design_bounds(self):
    return {name: (self.sigma_min, self.sigma_max) for name in PixelDensityDesign._fields}

  def combined_event_shape(self):
    return (self.n_grid, self.n_grid, 3)

  def size(self):
    return int(self.images.shape[0])

  # ------------------------------------------------------------------ #
  # Design scaling: log-affine over the sigma range, a factor being the natural step in a
  # concentration. The two axes share the range and are scaled independently.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    return (jnp.log(jnp.asarray(design, jnp.float32)) - self.log_sigma_min) / self.log_sigma_span

  def _to_nominal_flat(self, design_scaled):
    return jnp.exp(self.log_sigma_min + jnp.asarray(design_scaled, jnp.float32) * self.log_sigma_span)

  def sample_positions(self, design):
    """NOMINAL design -> ``(x, y)``, the probe positions along the COLUMN and ROW axes on the
    symmetric frame ``[-1, 1]``, each ``(..., n_grid)`` with the extremes pinned to ``-+1``.

    Equal probability mass of ``N(0, sigma^2)`` between consecutive probes, so the probe DENSITY is
    that Gaussian: small ``sigma`` collapses the interior onto the centre, large ``sigma`` flattens
    the Gaussian across the frame and the grid tends to uniform. Nothing ever clusters at the edge."""
    flat = jnp.asarray(self.flatten_design(design), jnp.float32)
    return self._positions(flat[..., 0]), self._positions(flat[..., 1])

  def _positions(self, sigma):
    """The quantiles of ``N(0, sigma^2)`` TRUNCATED to ``[-1, 1]``, at the stored levels.

    Written antisymmetrically about the centre -- ``Phi^-1(1/2 + d) = -Phi^-1(1/2 - d)`` -- so every
    inverse CDF is evaluated on the lower tail, where float32 has the precision, and the grid comes
    out exactly symmetric. The mass inside the truncation is ``erf(1 / (sigma * sqrt 2))`` rather
    than ``Phi(1/sigma) - Phi(-1/sigma)``, which cancels catastrophically at large ``sigma``. The
    argument is floored so no design can reach ``Phi^-1(0) = -inf``, and the two extreme probes are
    then overwritten by the truncation bounds they are BY DEFINITION."""
    sigma = sigma[..., None]
    span = jax.scipy.special.erf(HALF_WIDTH / (sigma * math.sqrt(2.0)))
    offset = (self.levels - 0.5) * span
    magnitude = -jax.scipy.special.ndtri(jnp.maximum(0.5 - jnp.abs(offset), TAIL_FLOOR))
    interior = sigma * jnp.sign(offset) * magnitude
    return jnp.where(self.edge_sign == 0.0, interior, self.edge_sign * HALF_WIDTH)

  def _weights(self, sigma, pixel_index, n_pixels):
    """Interpolation weights ``(..., n_grid, n_pixels)``: row ``i`` holds the cubic-convolution
    weights of grid point ``i`` over the pixel centres, with the four taps of Keys' kernel and zeros
    everywhere else. A tap outside the lattice has no column, which IS the zero padding."""
    positions = (self._positions(sigma) + HALF_WIDTH) * (0.5 * n_pixels / HALF_WIDTH) - 0.5
    return keys_kernel(positions[..., :, None] - pixel_index)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled, mask=None):
    """Raw ``MNISTEvent`` + SCALED design -> ``features (..., n_grid, n_grid, 3)``, channels-last.

    Channel 0 is the bicubic interpolant of the frame on ``[0, 1]`` read at the design's grid;
    channels 1 and 2 are the scaled ``sigma_x`` and ``sigma_y`` broadcast over it, which is how the
    design reaches the network -- the grid's spacing is not uniform and nothing else names the warp.
    Separable by construction: the row weights multiply the frame from the left and the column
    weights from the right, which is the 4x4 stencil written as two matrices. Differentiable in the
    scaled design. ``mask`` is unused -- every grid point is a real element."""
    image = jnp.asarray(event.image, jnp.float32) / 255.0
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:
      design_scaled = jnp.broadcast_to(design_scaled, image.shape[:-2] + design_scaled.shape)
    sigma = self._to_nominal_flat(design_scaled)
    row_weights = self._weights(sigma[..., 1], self.row_index, self.n_rows)
    column_weights = self._weights(sigma[..., 0], self.column_index, self.n_columns)
    samples = jnp.einsum('...ir,...rc,...jc->...ij', row_weights, image, column_weights)
    planes = jnp.broadcast_to(design_scaled[..., None, None, :], samples.shape + (2, ))
    return jnp.concatenate([samples[..., None], planes], axis=-1)

  def element_mask(self, event, mask):
    """All-ones over the GRID's rows: the combine's element axis is the grid row, and every grid
    point is a real reading."""
    return jnp.ones(jnp.asarray(event.image).shape[:-2] + (self.n_grid, ), jnp.int32)

  def normalize_target(self, target):
    """Identity on the one-hot label; the loss carries the scale."""
    flat, _ = tensor.flatten(target)
    return flat

  def denormalize_predictions(self, normalised):
    """Logits -> class probabilities."""
    probabilities = jax.nn.softmax(jnp.asarray(normalised, jnp.float32), axis=-1)
    return tensor.unflatten(tensor.structure(self.target_spec()), probabilities)

  def normalize_ground_truth(self, ground_truth):
    """The drawn digit's one-hot, flat."""
    flat, _ = tensor.flatten(ground_truth)
    return flat

  # ------------------------------------------------------------------ #
  # Objective
  # ------------------------------------------------------------------ #
  def loss_label(self):
    return f'cross-entropy, nats (digit; uniform = ln {self.n_classes})'

  def metric_labels(self):
    return ('loss', 'accuracy')

  def loss(self, predicted, target):
    """Per-sample softmax cross-entropy in nats, undivided, so uniform scores ``ln(n_classes)``."""
    return -jnp.sum(target * jax.nn.log_softmax(predicted, axis=-1), axis=-1)

  def metric(self, predicted, target):
    """Per-sample loss and a 0/1 accuracy indicator."""
    guess = jnp.argmax(predicted, axis=-1)
    truth = jnp.argmax(target, axis=-1)
    return {'loss': self.loss(predicted, target), 'accuracy': (guess == truth).astype(jnp.float32)}

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """The frames at ``event_index`` with their labels. DETERMINISTIC and DESIGN-INDEPENDENT: the
    grid is applied in :meth:`combine_scaled`, so an index is the same digit under every design and
    no design can move its own label. Indices wrap modulo :meth:`size`, matching the oversampling the
    shared event-index helper performs when a budget exceeds the data. ``design`` is accepted (and
    validated for width) to keep the contract's signature, not used."""
    event_index = np.asarray(event_index, np.int64)
    if event_index.ndim != 1:
      raise ValueError(f'event_index must be a flat array of indices, got shape {event_index.shape}')
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (-1, self.design_dim()))
    if flat.shape[0] not in (1, event_index.shape[0]):
      raise ValueError(f'design carries {flat.shape[0]} rows, which is neither 1 nor the {event_index.shape[0]} events')
    rows = event_index % self.size()
    digit = jnp.asarray(self.one_hot[rows])
    event = MNISTEvent(image=jnp.asarray(self.images[rows]))
    mask = jnp.ones((event_index.shape[0], self.n_rows), jnp.int32)
    return MNISTGroundTruth(digit=digit), event, mask, MNISTTarget(digit=digit)
