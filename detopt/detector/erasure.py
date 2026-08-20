"""Digit classification through random PIXEL ERASURE, over EMNIST or MNIST frames.

A dense, fixed-shape task: one 28x28 frame per event, no sparsity and no masking.

* **design** -- the keep probability ``p`` on ``[min_keep, 1]``, one coordinate. NOMINAL is ``p``;
  SCALED is ``1 - log(p) / log(min_keep)``, log-uniform, since the surviving ink spans decades.
* **event** -- the frame after erasure, ``(n_rows, n_columns)`` ``uint8``. Each pixel independently
  survives with probability ``p`` and is set to 0 otherwise; a surviving pixel shows its true
  intensity exactly.
* **combine** -- ``(n_rows, n_columns, 2)``: channel 0 is the surviving frame on ``[0, 1]``, channel
  1 the scaled keep probability broadcast over it. Both channels are bounded, so nothing is clipped,
  stabilised or renormalised anywhere.
* **target** -- the digit, one-hot over ``N_CLASSES``.
* **loss** -- softmax cross-entropy in nats, undivided. :meth:`design_penalty` prices the design at
  ``keep_weight * p``, added to the converged loss by the caller and never by the trainer; ``None``
  (no price configured) is distinct from ``0.0`` (a price that is zero).

WHAT MAKES IT HARD is that an ERASED pixel and a BACKGROUND pixel are both 0 and are genuinely
indistinguishable. The network loses a random ``1 - p`` share of the digit's ink with no way to tell
which share, so this is a subsampling problem rather than a signal-to-noise one. ``p = 1`` is pinned
at the top of the range, where the price is exactly ``ln(n_classes)`` -- the same pinning the loss
convention uses, and what makes the price commensurate with the cross-entropy it is added to.

The erasure key is ``fold_in(PRNGKey(event_index), bitcast_int32(p))``, so ``(design, event_index)``
reproduces a frame exactly, with nothing global and nothing call-order-dependent in it. A wrapped
index re-uses a digit under a fresh erasure, the key holding the unwrapped index.

⚠️ EMNIST frames are stored column-major and :func:`read_idx` transposes them; this detector must
NOT transpose again, the operation being its own inverse. Digits arrive through ``read_arrow``
(``data_path`` alone) or ``read_idx`` (plus ``labels_path``).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .common import Detector
from .mnist import N_CLASSES, MNISTGroundTruth, MNISTTarget, read_arrow, read_idx
from ..utils import tensor

__all__ = ['ErasureDetector', 'ErasureDesign', 'ErasureEvent']


class ErasureDesign(NamedTuple):
  """The keep probability, NOMINAL: one scalar on ``[min_keep, 1]``."""
  keep: jax.Array


class ErasureEvent(NamedTuple):
  """One frame after erasure, ``(n_rows, n_columns)`` ``uint8``; erased pixels are 0."""
  pixels: jax.Array


class ErasureDetector(Detector):
  """Digit frames seen through random pixel erasure (see the module docstring).

  Every constant is a constructor argument: the digit source, how many to read, the smallest keep
  probability the design range admits, and the price of a unit of keep probability.
  """

  def __init__(
    self, *, data_path: str, labels_path: str | None = None, n_events: int | None = None, min_keep: float = 1.0e-2,
    keep_weight: float | None = math.log(N_CLASSES)
  ):
    self.data_path = str(data_path)
    self.labels_path = None if labels_path is None else str(labels_path)
    self.n_events = None if n_events is None else int(n_events)
    self.min_keep = float(min_keep)
    self.keep_weight = None if keep_weight is None else float(keep_weight)
    self.n_classes = N_CLASSES
    if self.n_events is not None and self.n_events < 1:
      raise ValueError(f'n_events must be None or a positive int, got {n_events}')
    if not 0.0 < self.min_keep < 1.0:
      raise ValueError(
        f'min_keep is the bottom of a log range with 1.0 at the top, so it must lie '
        f'strictly inside (0, 1); got {min_keep}'
      )

    # `labels_path is None` selects the arrow layout, supplying one selects the IDX pair. Explicit
    # rather than sniffed from the extension: which loader ran decides whether frames were transposed.
    if self.labels_path is None:
      images, labels = read_arrow(self.data_path, self.n_events)
    else:
      images, labels = read_idx(self.data_path, self.labels_path, self.n_events)
    if images.ndim != 3:
      raise ValueError(f'{self.data_path} decoded to {images.shape}; expected (N, rows, columns) single-channel images')
    if labels.min() < 0 or labels.max() >= self.n_classes:
      raise ValueError(f'{self.data_path} holds labels in [{labels.min()}, {labels.max()}], outside [0, {self.n_classes})')
    # Held as uint8: the full EMNIST split is 188 MB this way and four times that as float32.
    self.images = images
    self.labels = labels
    self.n_rows = int(images.shape[1])
    self.n_columns = int(images.shape[2])
    self.one_hot = np.eye(self.n_classes, dtype=np.float32)[labels]
    # exp((1 - u) log min_keep) rather than min_keep * exp(u log(1/min_keep)), so u = 1 gives exactly 1.
    self.log_min_keep = math.log(self.min_keep)
    self._generate = jax.jit(jax.vmap(self._event))

  # ------------------------------------------------------------------ #
  # Record specs
  # ------------------------------------------------------------------ #
  def event_spec(self):
    return ErasureEvent(pixels=jax.ShapeDtypeStruct((self.n_rows, self.n_columns), np.uint8))

  def target_spec(self):
    return MNISTTarget(digit=jax.ShapeDtypeStruct((self.n_classes, ), np.float32))

  def ground_truth_spec(self):
    return MNISTGroundTruth(digit=jax.ShapeDtypeStruct((self.n_classes, ), np.float32))

  def design_shape(self):
    return (1, )

  def design_spec(self):
    return ErasureDesign(keep=jax.ShapeDtypeStruct((1, ), np.float32))

  def design_bounds(self):
    return {'keep': (self.min_keep, 1.0)}

  def combined_event_shape(self):
    return (self.n_rows, self.n_columns, 2)

  def size(self):
    return int(self.images.shape[0])

  # ------------------------------------------------------------------ #
  # Design scaling: log over the keep-probability decades.
  # ------------------------------------------------------------------ #
  def _to_scaled_flat(self, design):
    return 1.0 - jnp.log(jnp.asarray(design, jnp.float32)) / self.log_min_keep

  def _to_nominal_flat(self, design_scaled):
    return jnp.exp((1.0 - jnp.asarray(design_scaled, jnp.float32)) * self.log_min_keep)

  def design_penalty(self, design):
    """``keep_weight * p``, or ``None`` when no price is configured. Deterministic, so a caller adds
    it to the reported loss and leaves the reported error alone."""
    if self.keep_weight is None:
      return None
    return self.keep_weight * jnp.reshape(self.flatten_design(design), (-1, ))[0]

  # ------------------------------------------------------------------ #
  # Combine + normalisation
  # ------------------------------------------------------------------ #
  def combine_scaled(self, event, design_scaled, mask=None):
    """Raw ``ErasureEvent`` + SCALED design -> ``features (..., n_rows, n_columns, 2)``.

    Channel 0 is the surviving frame on ``[0, 1]``; channel 1 is the scaled keep probability, which
    the network needs because an erased pixel and a background pixel look identical and only the
    design says how much ink is missing. ``mask`` is unused -- every row of a dense frame is real."""
    pixels = jnp.asarray(event.pixels, jnp.float32) / 255.0
    design_scaled = jnp.asarray(design_scaled, jnp.float32)
    if design_scaled.ndim == 1:
      design_scaled = jnp.broadcast_to(design_scaled, pixels.shape[:-2] + design_scaled.shape)
    scaled = design_scaled[..., 0, None, None]
    return jnp.stack([pixels, jnp.broadcast_to(scaled, pixels.shape)], axis=-1)

  def element_mask(self, event, mask):
    """All-ones over the frame's rows: the combine's element axis is the row axis and a dense frame
    has nothing to mask out."""
    return jnp.ones(jnp.asarray(event.pixels).shape[:-1], jnp.int32)

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
    """Erase the frames at ``event_index`` under ``design`` (one keep probability, or one per event).
    Deterministic in ``(design, event_index)``; the image index wraps modulo :meth:`size` while the
    erasure key holds the unwrapped index, so a wrapped index gives a fresh erasure of the same
    digit."""
    event_index = np.asarray(event_index, np.int64)
    if event_index.ndim != 1:
      raise ValueError(f'event_index must be a flat array of indices, got shape {event_index.shape}')
    n = event_index.shape[0]
    flat = jnp.reshape(jnp.asarray(self.flatten_design(design), jnp.float32), (-1, self.design_dim()))
    if flat.shape[0] not in (1, n):
      raise ValueError(f'design carries {flat.shape[0]} rows, which is neither 1 nor the {n} events')
    keep = jnp.broadcast_to(flat[:, 0], (n, ))
    rows = event_index % self.size()
    pixels = self._generate(jnp.asarray(self.images[rows]), keep, jnp.asarray(event_index, jnp.int32))
    digit = jnp.asarray(self.one_hot[rows])
    mask = jnp.ones((n, self.n_rows), jnp.int32)
    return MNISTGroundTruth(digit=digit), ErasureEvent(pixels=pixels), mask, MNISTTarget(digit=digit)

  def _event(self, image, keep, event_index):
    """One frame: keep each pixel independently with probability ``keep``, erase the rest to 0."""
    key = jax.random.fold_in(jax.random.PRNGKey(event_index), jax.lax.bitcast_convert_type(keep, jnp.int32))
    survives = jax.random.bernoulli(key, keep, image.shape)
    return jnp.where(survives, image, jnp.zeros_like(image))
