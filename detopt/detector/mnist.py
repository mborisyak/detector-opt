"""MNIST digit classification through an optimisable visible WINDOW.

A dense, fixed-shape task: no sparsity, no masking of elements, every event the same shape. The
"detector" is a rectangular aperture over a 28x28 handwritten digit and the design is where that
aperture sits and how big it is; the network must name the digit from what the aperture exposes.

* **design** -- ``x = (x1, x2)`` and ``y = (y1, y2)``, every coordinate on ``[0, 1]``, flattening to
  ``(x1, x2, y1, y2)``. Each pair names two OPPOSITE edges in normalised image coordinates (``x``
  along the column axis, ``y`` along the row axis), so the window spans
  ``[min(x1, x2), max(x1, x2)) x [min(y1, y2), max(y1, y2))``. Every point of the unit box maps to a
  window wholly inside the image, so the design space carries NO constraints and no proposal can be
  infeasible. ⚠️ Each pair is UNORDERED and the two pairs swap independently, so the objective is
  invariant under a group of order 4 and a quarter of the cube already holds every window there is --
  which is why the surrogate should use the ``independent-sorting-rbf`` kernel, whose sort
  canonicalises a design to ``(left, right, top, bottom)`` before the RBF ever sees it. NOMINAL and
  SCALED are the same space and the bijection between them is the identity.
* **event** -- one MNIST image, ``(28, 28)`` ``uint8``, read from the HuggingFace arrow file at
  construction. The event does NOT depend on the design: the aperture is applied in
  :meth:`combine_scaled`, so one pool of images serves every design and no design can move its own
  label.
* **combine** -- ``(28, 28, 2)`` float32: the image OCCLUDED by the window, beside the binary window
  mask itself. Occluding is not optional -- an intact image channel would let the network read the
  whole digit and the design would buy nothing. The mask channel is what carries the design into the
  network, and it is needed even though the image is already occluded: without it a black pixel
  INSIDE the window is indistinguishable from a pixel outside it, so the network could not tell a
  small window from a large one over blank paper.
* **target** -- the digit, one-hot over 10 classes.
* **loss** -- softmax cross-entropy in NATS, undivided. Uniform guessing scores ``ln 10 = 2.303``.

THE PRICE OF LOOKING
--------------------
With nothing to pay, the best window is trivially the whole image and there is no design problem
left. :meth:`design_penalty` therefore prices the VISIBLE AREA -- the image fraction
``w (1 - x) h (1 - y)``, on ``[0, 1]`` -- at ``area_weight`` loss units per unit area, and
``scripts/bo.py`` adds it to the converged loss AFTER training. It never reaches the trainer: the
convergence criterion judges how well the network fits, and a per-design constant added there would
move the reported loss without changing anything that criterion measures.

``area_weight`` defaults to ``ln(n_classes)``, which is exactly the cross-entropy of a uniform guess.
At that weight SEEING THE WHOLE IMAGE COSTS PRECISELY AS MUCH AS KNOWING NOTHING ABOUT THE DIGIT, so
the two degenerate extremes are priced identically -- a full-image window pays ``ln 10`` in area and
~0 in cross-entropy, a vanishing window pays ~0 in area and ``ln 10`` in cross-entropy -- and
``ln 10`` becomes a reference line on the reported total: above it, the design is worse than useless.
The value follows from the number of classes rather than from tuning. The weight stays configurable
so that trade-off can be moved; ``area_weight: null`` removes the price ENTIRELY and
:meth:`design_penalty` then returns ``None``, which is the pure-classification debugging mode. A
weight of ``0.0`` is a different thing: a price that was measured and came out zero.
"""

import io
import math
import struct
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import Detector
from ..utils import tensor

__all__ = ['MNISTDetector', 'MNISTDesign', 'MNISTEvent', 'MNISTTarget', 'MNISTGroundTruth', 'N_CLASSES', 'read_arrow', 'read_idx']

N_CLASSES = 10


class MNISTDesign(NamedTuple):
  """The visible window, NOMINAL, as two UNORDERED PAIRS of opposite edges: ``x = (x1, x2)`` on the
  column axis and ``y = (y1, y2)`` on the row axis, in normalised image coordinates. Each field is
  ``(2,)`` and each coordinate lives on ``[0, 1]``; the flat design is ``(x1, x2, y1, y2)``.

  Which member of a pair is the smaller does not matter, since :meth:`MNISTDetector.window` sorts
  them -- so the map is 2-to-1 per axis and the objective is invariant under a group of order 4. The
  pair is the FIELD rather than four scalars precisely so that symmetry is declarable: a kernel
  asking for blocks of 2 finds ``x`` and ``y`` and nothing else."""
  x: jax.Array
  y: jax.Array


class MNISTEvent(NamedTuple):
  """The whole digit image, ``(n_rows, n_columns)`` ``uint8``. The window is applied in
  :meth:`MNISTDetector.combine_scaled`, never here."""
  image: jax.Array


class MNISTTarget(NamedTuple):
  """What the regressor predicts: the digit, one-hot over ``N_CLASSES``."""
  digit: jax.Array


class MNISTGroundTruth(NamedTuple):
  """The drawn digit (== conditioning), one-hot over ``N_CLASSES``. The target is the whole of it
  here -- an MNIST image has no hidden generative parameters to condition on."""
  digit: jax.Array


def read_arrow(path, n_events):
  """Decode a HuggingFace MNIST arrow file into ``(images uint8 (N, H, W), labels int32 (N,))``.

  The ``image`` column holds PNG bytes, so every row is decoded ONCE, here, into one dense array.
  ``n_events`` (or ``None``) caps how many leading rows are read."""
  import pyarrow
  from PIL import Image

  with pyarrow.memory_map(str(path), 'rb') as source:
    table = pyarrow.ipc.open_stream(source).read_all()
  missing = {'image', 'label'} - set(table.column_names)
  if len(missing) > 0:
    raise ValueError(f'{path} has columns {table.column_names}, missing {sorted(missing)}')
  if n_events is not None:
    table = table.slice(0, int(n_events))
  labels = np.asarray(table.column('label').to_pylist(), np.int32)
  images = np.stack([np.asarray(Image.open(io.BytesIO(row['bytes'])), np.uint8) for row in table.column('image').to_pylist()])
  return images, labels


def read_idx(images_path, labels_path, n_events):
  """Decode an IDX image/label pair into ``(images uint8 (N, H, W), labels int32 (N,))``.

  The format the raw MNIST and EMNIST distributions ship in: a big-endian header (magic, count, and
  for images rows and columns) followed by raw bytes. ``n_events`` (or ``None``) caps how many
  leading records are read.

  ⚠️ EVERY IMAGE IS TRANSPOSED on the way out. The EMNIST authors store their frames column-major
  relative to MNIST, so the raw bytes decode to a digit lying on its side and mirrored; ``.T`` puts
  it upright. That is not cosmetic here -- the design IS a rectangle in image coordinates, so an
  uncorrected transpose would silently swap the roles of the two axes and make `left`/`right` mean
  `top`/`bottom`. Verified by eye on a chiral digit rather than assumed. The transpose is its own
  inverse, so applying it to genuine MNIST IDX files would break them instead; those come through
  :func:`read_arrow`."""
  def _read(path, magic_expected, n_header_fields):
    with open(path, 'rb') as f:
      header = f.read(4 * n_header_fields)
      fields = struct.unpack(f'>{n_header_fields}I', header)
      if fields[0] != magic_expected:
        raise ValueError(f'{path} has IDX magic {fields[0]}, expected {magic_expected}')
      count = int(fields[1])
      taken = count if n_events is None else min(count, int(n_events))
      per_record = int(np.prod(fields[2:])) if len(fields) > 2 else 1
      return taken, np.frombuffer(f.read(taken * per_record), np.uint8), fields

  n_images, image_bytes, image_fields = _read(images_path, 2051, 4)
  n_labels, label_bytes, _ = _read(labels_path, 2049, 2)
  if n_images != n_labels:
    raise ValueError(f'{images_path} holds {n_images} images but {labels_path} holds {n_labels} labels')
  rows, columns = int(image_fields[2]), int(image_fields[3])
  images = image_bytes.reshape(n_images, rows, columns).transpose(0, 2, 1)
  return np.ascontiguousarray(images), label_bytes.astype(np.int32)


class MNISTDetector(Detector):
  """A rectangular visible window over an MNIST digit (see the module docstring).

  Every constant is a constructor argument, i.e. lives in the yaml config: which file the digits come
  from, how many of them to read, and what a unit of visible area costs. Two layouts are supported --
  a HuggingFace arrow file (``data_path`` alone) and an IDX image/label pair (``data_path`` plus
  ``labels_path``, which is how the raw EMNIST digits split is read).
  """

  def __init__(
    self, *, data_path: str, labels_path: str | None = None, n_events: int | None = None,
    area_weight: float | None = math.log(N_CLASSES)
  ):
    self.data_path = str(data_path)
    self.labels_path = None if labels_path is None else str(labels_path)
    self.n_events = None if n_events is None else int(n_events)
    self.area_weight = None if area_weight is None else float(area_weight)
    if self.n_events is not None and self.n_events < 1:
      raise ValueError(f'n_events must be None or a positive int, got {n_events}')
    self.n_classes = N_CLASSES

    # `labels_path is None` selects the single-file HuggingFace arrow layout; supplying one selects
    # the two-file IDX layout the raw MNIST and EMNIST distributions ship in. Explicit rather than
    # sniffed from the extension: which loader ran decides whether the frames were transposed.
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
    # Pixel CENTRES in normalised coordinates: pixel k covers [k/n, (k+1)/n), so its centre decides
    # whether the pixel is inside the window.
    self.row_centres = jnp.asarray((np.arange(self.n_rows) + 0.5) / self.n_rows, jnp.float32)
    self.column_centres = jnp.asarray((np.arange(self.n_columns) + 0.5) / self.n_columns, jnp.float32)

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
    return (4, )

  def design_spec(self):
    pair = jax.ShapeDtypeStruct((2, ), np.float32)
    return MNISTDesign(x=pair, y=pair)

  def design_bounds(self):
    return {name: (0.0, 1.0) for name in MNISTDesign._fields}

  def combined_event_shape(self):
    return (self.n_rows, self.n_columns, 2)

  def size(self):
    return int(self.images.shape[0])

  # ------------------------------------------------------------------ #
  # Design scaling: the IDENTITY. The parameterisation is already the unit box -- every coordinate
  # is a fraction, of the image for the corner and of the remaining space for the extent -- so there
  # is nothing to map and NOMINAL == SCALED coordinate by coordinate.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    return jnp.asarray(design, jnp.float32)

  def _to_nominal_flat(self, design_scaled):
    return jnp.asarray(design_scaled, jnp.float32)

  def window(self, design_scaled):
    """SCALED design ``(..., 4)`` -> the window's half-open extent ``(x_lo, x_hi, y_lo, y_hi)`` in
    normalised image coordinates, each ``(...,)``.

    The design names two opposite CORNERS and this sorts them: ``x_lo = min(x1, x2)``,
    ``x_hi = max(x1, x2)``, likewise for ``y``. Every point of the unit box therefore maps to a window
    wholly inside the image with no constraint on the design space, and the ordering of a pair carries
    no meaning."""
    d = jnp.asarray(design_scaled, jnp.float32)
    x1, x2, y1, y2 = d[..., 0], d[..., 1], d[..., 2], d[..., 3]
    return jnp.minimum(x1, x2), jnp.maximum(x1, x2), jnp.minimum(y1, y2), jnp.maximum(y1, y2)

  def visible_area(self, design):
    """NOMINAL design -> the window's area as a FRACTION of the image,
    ``|x1 - x2| * |y1 - y2|`` on ``[0, 1]``. What :meth:`design_penalty` prices."""
    x0, x1, y0, y1 = self.window(self.to_scaled(design))
    return (x1 - x0) * (y1 - y0)

  def design_penalty(self, design):
    """``area_weight * visible_area(design)``, or ``None`` when ``area_weight`` is ``None``.

    Deterministic in the design, so a caller adds it to the reported loss and leaves the reported
    error alone. ``None`` (no price configured) and ``0.0`` (a price that is zero) are different
    answers and the caller must keep them apart."""
    if self.area_weight is None:
      return None
    return self.area_weight * self.visible_area(design)

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled, mask=None):
    """Raw ``MNISTEvent`` + SCALED design -> ``features (..., n_rows, n_columns, 2)``, channels-last.

    Channel 0 is the image on ``[0, 1]`` MULTIPLIED by the window, so nothing outside the aperture
    reaches the network; channel 1 is the binary window itself, which is how the design enters and
    what separates "dark inside the window" from "outside it". A pixel is inside when its CENTRE is,
    on the half-open extent, so ``w = 0`` or ``h = 0`` shows nothing at all. ``mask`` is unused:
    every row of a dense image is a real element."""
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    image = jnp.asarray(event.image, jnp.float32) / 255.0
    if design_scaled.ndim == 1:
      design_scaled = jnp.broadcast_to(design_scaled, image.shape[:-2] + design_scaled.shape)
    x0, x1, y0, y1 = self.window(design_scaled)
    in_columns = (self.column_centres >= x0[..., None]) & (self.column_centres < x1[..., None])
    in_rows = (self.row_centres >= y0[..., None]) & (self.row_centres < y1[..., None])
    window = (in_rows[..., :, None] & in_columns[..., None, :]).astype(jnp.float32)
    return jnp.stack([image * window, window], axis=-1)

  def element_mask(self, event, mask):
    """Every image ROW is a valid element -- the combine's element axis is the row axis and a dense
    image has nothing to mask out."""
    return jnp.ones(jnp.asarray(event.image).shape[:-1], jnp.int32)

  def normalize_target(self, target):
    """IDENTITY on the one-hot label: there is no scale to remove. The normalisation lives in the
    LOSS, which reads in nats against the fixed reference ``ln(n_classes)``."""
    flat, _ = tensor.flatten(target)
    return flat

  def denormalize_predictions(self, normalised):
    """Logits -> class PROBABILITIES, the physical reading of a digit call."""
    probabilities = jax.nn.softmax(jnp.asarray(normalised, jnp.float32), axis=-1)
    return tensor.unflatten(tensor.structure(self.target_spec()), probabilities)

  def normalize_ground_truth(self, ground_truth):
    """The drawn digit's one-hot, flat. Already on ``[0, 1]`` -- nothing to standardise."""
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
    """Per-sample softmax cross-entropy over the ``N_CLASSES`` logits, in NATS and UNDIVIDED, so the
    uniform prediction scores ``ln(n_classes)`` and the area price is on the same scale."""
    return -jnp.sum(target * jax.nn.log_softmax(predicted, axis=-1), axis=-1)

  def metric(self, predicted, target):
    """Per-sample loss and accuracy (a 0/1 indicator, so its mean is the accuracy)."""
    guess = jnp.argmax(predicted, axis=-1)
    truth = jnp.argmax(target, axis=-1)
    return {'loss': self.loss(predicted, target), 'accuracy': (guess == truth).astype(jnp.float32)}

  # ------------------------------------------------------------------ #
  # Event generation
  # ------------------------------------------------------------------ #
  def __call__(self, design, event_index):
    """The images at ``event_index``, with their labels. DETERMINISTIC and DESIGN-INDEPENDENT: the
    aperture is applied in :meth:`combine_scaled`, so the same index is the same digit under every
    design and no design can move its own label. Indices wrap modulo :meth:`size`, matching the
    oversampling the shared event-index helper performs when a budget exceeds the data. ``design``
    is accepted (and validated for width) to keep the contract's signature, not used."""
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
