"""GP kernels that know about a design's exchangeability.

The enzyme design is a BATCH of experiments and the regressor aggregates over them
permutation-invariantly, so all ``m!`` orderings of one design are the same design with the same
loss. A plain ARD-RBF does not know that: it models ``m!`` copies of every optimum, searches ``m!``
times more volume than exists, and -- worse -- spends one lengthscale per experiment SLOT on a
distinction the problem cannot make.

:class:`PermutationInvariantRBF` builds the invariance into the kernel itself,

    k(x, x') = (1/|G|) sum_pi  sigma^2 exp(-1/2 sum_g |x_g - (pi.x')_g|^2 / l_g^2),

by averaging an equivariant base over the group. Two consequences worth being explicit about:

* **Lengthscales are tied within an exchangeable block, and that is the point, not a cost.** If the
  function is symmetric in the experiments then "experiment 3's temperature has a shorter
  lengthscale than experiment 1's" is not a statement the problem can make. Blocks that are NOT
  permuted into each other keep their own lengthscale, so the ARD that means something -- here
  ``temperature`` against ``enzyme_fraction`` -- is untouched. For the enzyme design this is 8
  lengthscales down to 2, fitted from ~60 observations: a far better-posed marginal likelihood, and
  a direct answer to the measured pathology of ARD lengthscales pinning at the prior ceiling.

* **With no exchangeable block it degrades exactly to ARD-RBF** (``|G| = 1``, one lengthscale per
  coordinate) -- a property, verified by a test, and NOT how a design without a symmetry should be
  served. The straw/stereo geometries are not symmetric (a layer's wire stagger is keyed to its
  index parity, and the stereo stations are already ordered by their coupled window), so they take
  :class:`ARDRBF`, which is sklearn's plain ``ConstantKernel * RBF`` and says what it models.

Not stationary (``k(x, x')`` is not a function of ``x - x'`` once the group is averaged over) and
``diag`` is not constant, so neither of sklearn's stationary mixins applies.

PRIOR BOUNDS. Every production path supplies its own, from the config's ``gp.log_*_prior_bounds``
through :func:`detopt.bo.kernel_from_config` or :class:`~detopt.bo.BayesianOptimizer`, and that is
where a bound belongs: it is a statement about the TASK's scales, not about the kernel. The
constructor defaults below serve only a kernel built by hand, and they are:

* ``length_scale_bounds`` FROM THE BOX. These kernels are only ever evaluated on the scaled cube
  ``[0, 1]^d``, so a coordinate's full range is 1 and both ends have a meaning. At the ceiling the
  correlation across that whole range is ``exp(-1/(2 l^2))``, within half a percent of 1 -- the
  coordinate is switched off, and a longer lengthscale is not distinguishable from it. At the floor
  the lengthscale is a hundredth of the range, below the spacing any run of tens of designs has, so
  the GP already interpolates its own observations and a shorter one buys nothing.
* ``constant_value_bounds`` from NOTHING here. The amplitude is a prior variance in the OBJECTIVE's
  units, which a kernel cannot know -- the driver centres ``y`` but never scales it -- so this
  default is a bracket, not a derivation, and a task that cares must state its own.
"""

import itertools
import math

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Hyperparameter, Kernel

__all__ = ["ARDRBF", "PermutationInvariantRBF", "NormalisedInvariantRBF", "SortingRBF"]


class ARDRBF(Kernel):
  """``amplitude^2 * RBF(l)``: sklearn's OWN ``ConstantKernel * RBF``, with the three things the EI
  optimiser needs bolted on -- the design dimension ``d``, ``k_and_grad_x`` (the gradient in the
  INPUT, which sklearn does not expose) and ``grad_diag``.

  This is the no-symmetry surrogate, and it is its own class deliberately. The same numbers come out
  of :class:`PermutationInvariantRBF` with ``blocks=()``, but by a route that averages over a group
  of one and takes the permanent of a 1x1 matrix -- a modelling statement ("this design is a set")
  written in the code for designs that are not, and a Ryser pass to say nothing. Naming the plain
  kernel keeps "no symmetry" a CHOICE in the registry rather than a degenerate case of the other.

  The maths is delegated: :meth:`__call__` and :meth:`diag` are sklearn's product, so the ARD path
  cannot drift away from the library's own implementation of it.
  """

  def __init__(
    self, d, constant_value=1.0, constant_value_bounds=(1e-6, 1e2), length_scale=1.0, length_scale_bounds=(1e-2, 1e1)
  ):
    self.d = d
    self.constant_value = constant_value
    self.constant_value_bounds = constant_value_bounds
    # Broadcast a scalar to one entry per COORDINATE here: sklearn reads `n_elements` off
    # `hyperparameter_length_scale` but builds `theta` from the ATTRIBUTE, so a scalar leaves theta
    # and bounds different lengths and `fit` dies with "The number of bounds is not compatible with
    # the length of x0".
    self.length_scale = length_scale
    self.length_scale_bounds = length_scale_bounds
    if np.atleast_1d(np.asarray(length_scale, dtype=float)).size == 1 and d > 1:
      self.length_scale = np.full(d, float(np.ravel(length_scale)[0]))

  @property
  def n_length_scales(self):
    return self.d

  # sklearn hyperparameter plumbing (names must match the __init__ args). `hyperparameters` is
  # collected by `dir()`, i.e. alphabetically, so theta is [log amplitude^2, log l] -- the same
  # order sklearn's own Product produces, which is what lets `__call__` forward its gradient
  # columns unchanged.
  @property
  def hyperparameter_constant_value(self):
    return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)

  @property
  def hyperparameter_length_scale(self):
    return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, self.d)

  @property
  def _kernel(self):
    """sklearn's ``ConstantKernel * RBF`` at the CURRENT hyperparameters.

    Built per call, not stored: sklearn tunes a kernel by ``clone_with_theta``, which assigns the
    attributes directly and never re-runs ``__init__``, so anything cached from the constructor
    would silently answer with the hyperparameters the fit started from."""
    return (
      ConstantKernel(float(self.constant_value), self.constant_value_bounds) *
      RBF(self.coordinate_length_scales(), self.length_scale_bounds)
    )

  def coordinate_length_scales(self):
    """The lengthscale of each DESIGN COORDINATE, ``(d,)``. Same name and meaning as the invariant
    kernels', where it repeats a block's shared value -- here every coordinate has its own."""
    length_scale = np.atleast_1d(np.asarray(self.length_scale, dtype=float))
    if length_scale.size == 1:
      length_scale = np.full(self.d, length_scale[0])
    if length_scale.size != self.d:
      raise ValueError(f"length_scale has {length_scale.size} entries, expected {self.d}")
    return length_scale

  def __call__(self, X, Y=None, eval_gradient=False):
    return self._kernel(np.atleast_2d(X), None if Y is None else np.atleast_2d(Y), eval_gradient)

  def diag(self, X):
    return self._kernel.diag(np.atleast_2d(X))

  def grad_diag(self, x):
    """Exactly zero: ``k(x, x) = amplitude^2`` for every ``x``, so the posterior variance's prior
    term does not move with the proposal."""
    return np.zeros(self.d)

  def k_and_grad_x(self, X_train, x):
    """``k(x, X_train)`` and ``dk/dx``: ``(n,)`` and ``(n, d)``. The acquisition needs the gradient
    in the INPUT, which no sklearn kernel exposes, so this one expression is ours."""
    x = np.asarray(x, dtype=float).ravel()
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    length_scale = self.coordinate_length_scales()
    difference = X_train - x[None, :]  # (n, d)
    exponent = (difference * difference / (length_scale**2)[None, :]).sum(axis=1)
    k = float(self.constant_value) * np.exp(-0.5 * exponent)  # (n,)
    return k, k[:, None] * difference / (length_scale**2)[None, :]

  def is_stationary(self):
    return True

  def __repr__(self):
    return (
      f"{type(self).__name__}(d={self.d}, amplitude^2={float(self.constant_value):.3g}, "
      f"l={np.round(self.coordinate_length_scales(), 3)})"
    )


class PermutationInvariantRBF(Kernel):
  """``amplitude^2 * RBF``, averaged over the permutations of a set of exchangeable blocks.

  ``blocks`` is a tuple of equal-length index tuples, all permuted by the SAME permutation, because
  they are different FIELDS of one exchangeable element -- for the enzyme batch, experiment ``k``
  is the pair ``(enzyme_fraction[k], temperature[k])``, so the two blocks move together. Every
  coordinate not named in a block is free and carries its own lengthscale.

  The lengthscale vector is laid out as ``[one per block] + [one per free coordinate]``.
  """

  def __init__(
    self, d, blocks=(), constant_value=1.0, constant_value_bounds=(1e-6, 1e2), length_scale=1.0,
    length_scale_bounds=(1e-2, 1e1)
  ):
    self.d = d
    self.blocks = blocks
    self.constant_value = constant_value
    self.constant_value_bounds = constant_value_bounds
    # Broadcast a scalar to one entry per GROUP here. sklearn reads `n_elements` off
    # `hyperparameter_length_scale` (= n_length_scales) but builds `theta` from the ATTRIBUTE, so a
    # scalar leaves theta and bounds different lengths and `fit` dies with "The number of bounds is
    # not compatible with the length of x0". Both call sites used to paper over this by reassigning
    # the attribute after construction.
    self.length_scale = length_scale
    self.length_scale_bounds = length_scale_bounds
    if np.atleast_1d(np.asarray(length_scale, dtype=float)).size == 1 and self.n_length_scales > 1:
      self.length_scale = np.full(self.n_length_scales, float(np.ravel(length_scale)[0]))

  # ------------------------------------------------------------------ #
  # Layout: which coordinates share a lengthscale, and what the group is.
  # ------------------------------------------------------------------ #
  @property
  def _groups(self):
    """``(block_index_arrays, free_index_array)`` -- the coordinate groups sharing a lengthscale.

    Cached: this is called once per permutation per kernel evaluation, and it depends only on
    ``d``/``blocks``, which sklearn never tunes (it re-clones the kernel to change hyperparameters,
    so the cache cannot go stale)."""
    cached = getattr(self, "_groups_cache", None)
    if cached is None:
      blocks = [np.asarray(b, dtype=int) for b in self.blocks]
      used = np.concatenate(blocks) if len(blocks) > 0 else np.empty(0, dtype=int)
      cached = (blocks, np.setdiff1d(np.arange(self.d), used))
      self._groups_cache = cached
    return cached

  @property
  def n_length_scales(self):
    blocks, free = self._groups
    return len(blocks) + free.size

  @property
  def _permutations(self):
    """The group, as position permutations applied simultaneously to every block. Empty ``blocks``
    (or a block of one) leaves the identity, i.e. an ordinary ARD-RBF."""
    blocks, _ = self._groups
    if len(blocks) == 0:
      return [()]
    return list(itertools.permutations(range(blocks[0].size)))

  # ------------------------------------------------------------------ #
  # sklearn hyperparameter plumbing (names must match the __init__ args).
  # ------------------------------------------------------------------ #
  @property
  def hyperparameter_constant_value(self):
    return Hyperparameter("constant_value", "numeric", self.constant_value_bounds)

  @property
  def hyperparameter_length_scale(self):
    return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, self.n_length_scales)

  # ------------------------------------------------------------------ #
  def _length_scales(self):
    """The per-GROUP lengthscales as a ``(n_groups,)`` array (a scalar broadcasts)."""
    length_scale = np.atleast_1d(np.asarray(self.length_scale, dtype=float))
    if length_scale.size == 1:
      length_scale = np.full(self.n_length_scales, length_scale[0])
    if length_scale.size != self.n_length_scales:
      raise ValueError(f"length_scale has {length_scale.size} entries, expected {self.n_length_scales}")
    return length_scale

  def _permuted(self, Y, order):
    """``Y`` with every exchangeable block reordered by ``order`` (identity when there is no group)."""
    blocks, _ = self._groups
    if len(order) == 0:
      return Y
    out = np.array(Y, copy=True)
    for block in blocks:
      out[:, block] = Y[:, block[list(order)]]
    return out

  @staticmethod
  def _permanent_and_derivatives(A, weights):
    """``perm(A)`` and ``d perm(A) / d theta_b`` for each ``weights[b]``, in ONE Ryser pass.

    The entries depend on the hyperparameters as ``dA_ij/dtheta_b = A_ij * W^b_ij``, and rather than
    differentiate the permanent's DEFINITION we differentiate Ryser's formula. With
    ``R_i(S) = sum_{j in S} A_ij`` and ``P(S) = prod_i R_i(S)``,

        d/dtheta_b  P(S) = sum_i ( prod_{i' != i} R_i'(S) ) * dR_i(S),

    so the derivative rides along the same subset loop at the same ``O(2^m m)``. Differentiating the
    definition instead gives ``sum_kl perm(A^(k,l)) A_kl W_kl`` -- ``m^2`` MINOR permanents, which is
    worse than brute force below ``m ~ 7`` and needs a second algorithm besides.

    The product-of-the-OTHERS is formed from prefix and suffix cumulative products rather than as
    ``P(S) / R_i(S)``. The entries are RBF values, so ``R_i(S)`` is positive in exact arithmetic --
    but it UNDERFLOWS to exactly 0 in float64 once a lengthscale is short enough, which the prior
    bounds (``log l`` down to -6, i.e. ``l = 0.0025`` against unit-cube distances) allow and which
    L-BFGS reaches on its own restarts. Dividing there produced a NaN gradient, L-BFGS gave up at
    iteration 0 with ``ABNORMAL``, and the kernel silently kept its INITIAL hyperparameters -- a fit
    that never happened, reported as a fit. The prefix/suffix form has no division, is exact when a
    row vanishes, and costs the same at these ``m``."""
    m = A.shape[-1]
    total = np.zeros(A.shape[:-2])
    derivatives = [np.zeros(A.shape[:-2]) for _ in weights]
    for mask in range(1, 1 << m):
      columns = [j for j in range(m) if (mask >> j) & 1]
      row_sums = A[..., :, columns].sum(axis=-1)  # R_i(S), (..., m)
      product = np.prod(row_sums, axis=-1)  # P(S)
      sign = (-1.0)**len(columns)
      total += sign * product
      if len(weights) == 0:
        continue
      # prod_{i' != i} R_i'(S), as (everything before i) * (everything after i).
      ones = np.ones(A.shape[:-2] + (1, ))
      prefix = np.concatenate([ones, np.cumprod(row_sums[..., :-1], axis=-1)], axis=-1)
      suffix = np.concatenate([np.cumprod(row_sums[..., :0:-1], axis=-1)[..., ::-1], ones], axis=-1)
      others = prefix * suffix
      for b, weight in enumerate(weights):
        row_derivative = (A[..., :, columns] * weight[..., :, columns]).sum(axis=-1)  # dR_i(S)
        derivatives[b] += sign * (row_derivative * others).sum(axis=-1)
    parity = (-1.0)**m
    return parity * total, [parity * d for d in derivatives]

  @staticmethod
  def _permanent(A):
    """``perm(A)`` for a stack of ``(..., m, m)`` matrices, by RYSER's formula.

    ``perm(A) = (-1)^m sum_{S subset of columns} (-1)^{|S|} prod_i sum_{j in S} A_ij``, which is
    ``O(2^m m)`` against the ``O(m! m)`` of summing over permutations explicitly. At ``m = 4`` that
    is 64 operations against 96 -- no reason to care -- but it is 2048 against 322560 at ``m = 8``,
    so the group can grow with ``n_experiments`` without the kernel becoming the bottleneck.
    (Permanents are #P-hard: this is a much better exponential, not a polynomial one.)"""
    m = A.shape[-1]
    total = np.zeros(A.shape[:-2])
    for mask in range(1, 1 << m):
      columns = [j for j in range(m) if (mask >> j) & 1]
      row_sums = A[..., :, columns].sum(axis=-1)  # (..., m)
      total += (-1.0)**len(columns) * np.prod(row_sums, axis=-1)
    return (-1.0)**m * total

  def _pair_terms(self, X, Y):
    """The pieces every path needs, each computed ONCE per point pair.

    Returns ``(free_factor, element, block_distance)``:

    * ``free_factor`` ``(n_X, n_Y)`` -- the RBF over coordinates that no permutation touches;
    * ``element`` ``(n_X, n_Y, m, m)`` -- ``A[i,j,k,l]``, the RBF between element ``k`` of ``x_i``
      and element ``l`` of ``y_j``, with the blocks multiplied together because they permute as one
      (experiment ``k`` IS the ``(fraction, temperature)`` pair). ``None`` when there is no group;
    * ``block_distance`` ``(n_blocks, n_X, n_Y, m, m)`` -- each block's scaled squared distance,
      kept for the lengthscale gradient.

    This is the point of the rewrite: the group average needs only these ``m^2`` elementwise
    exponentials per pair, not one full d-dimensional distance per permutation."""
    blocks, free = self._groups
    length_scale = self._length_scales()

    free_factor = np.ones((X.shape[0], Y.shape[0]))
    for j, coordinate in enumerate(free):
      difference = X[:, coordinate][:, None] - Y[:, coordinate][None, :]
      free_factor = free_factor * np.exp(-0.5 * difference * difference / length_scale[len(blocks) + j]**2)
    if len(blocks) == 0:
      return free_factor, None, None

    block_distance = []
    exponent = np.zeros((X.shape[0], Y.shape[0], blocks[0].size, blocks[0].size))
    for i, block in enumerate(blocks):
      difference = X[:, block][:, None, :, None] - Y[:, block][None, :, None, :]  # (nX, nY, m, m)
      scaled = difference * difference / length_scale[i]**2
      block_distance.append(scaled)
      exponent = exponent + scaled
    return free_factor, np.exp(-0.5 * exponent), np.stack(block_distance)

  def __call__(self, X, Y=None, eval_gradient=False):
    X = np.atleast_2d(X)
    symmetric = Y is None
    Y = X if symmetric else np.atleast_2d(Y)
    if eval_gradient and not symmetric:
      raise ValueError("gradient can only be evaluated when Y is None")

    amplitude = float(self.constant_value)
    blocks, free = self._groups
    length_scale = self._length_scales()
    free_factor, element, block_distance = self._pair_terms(X, Y)

    grad_length = np.zeros((X.shape[0], Y.shape[0], length_scale.size))
    if element is None:  # no exchangeable group: an ordinary ARD-RBF
      total = amplitude * free_factor
    else:
      # The group average IS the permanent of the elementwise matrix, over m!. Value and gradient
      # come out of the SAME Ryser pass, so they cannot disagree.
      m = element.shape[-1]
      if eval_gradient:
        permanent, derivatives = self._permanent_and_derivatives(element, list(block_distance))
      else:
        permanent, derivatives = self._permanent(element), []
      scale = amplitude * free_factor / float(math.factorial(m))
      total = scale * permanent
      if eval_gradient:
        for i in range(len(blocks)):
          grad_length[:, :, i] = scale * derivatives[i]

    if eval_gradient:
      # Free coordinates factorise out of the permanent entirely: d/dlog(l_c) exp(-s_c/2) = k * s_c.
      for j, coordinate in enumerate(free):
        difference = X[:, coordinate][:, None] - Y[:, coordinate][None, :]
        column = len(blocks) + j
        grad_length[:, :, column] = total * (difference * difference / length_scale[column]**2)

    if not eval_gradient:
      return total
    # theta is log(constant_value) and log(length_scale); k is linear in constant_value, so
    # d k / d log(constant_value) = k.
    gradient = [total[:, :, None]] if not self.hyperparameter_constant_value.fixed else []
    if not self.hyperparameter_length_scale.fixed:
      gradient.append(grad_length)
    return total, (np.dstack(gradient) if len(gradient) > 0 else np.empty((X.shape[0], X.shape[0], 0)))

  def coordinate_length_scales(self):
    """The lengthscale of each DESIGN COORDINATE, ``(d,)`` -- every coordinate of an exchangeable
    block repeats its block's value. This is what an x-gradient needs."""
    blocks, free = self._groups
    length_scale = self._length_scales()
    out = np.empty(self.d)
    for i, block in enumerate(blocks):
      out[block] = length_scale[i]
    for j, coordinate in enumerate(free):
      out[coordinate] = length_scale[len(blocks) + j]
    return out

  def k_and_grad_x(self, X_train, x):
    """``k(x, X_train)`` and ``dk/dx``: ``(n,)`` and ``(n, d)``.

    The acquisition needs the gradient in the INPUT, not in the hyperparameters, and for a
    group-averaged kernel it is the average of the per-permutation gradients -- NOT the ARD-RBF
    expression at some effective distance. Kept with the kernel that defines it, so the EI optimiser
    cannot silently assume a form the kernel does not have."""
    x = np.asarray(x, dtype=float).ravel()
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    amplitude = float(self.constant_value)
    per_coordinate = self.coordinate_length_scales()

    total_k = np.zeros(X_train.shape[0])
    total_jac = np.zeros(X_train.shape)
    for order in self._permutations:
      permuted = self._permuted(X_train, order)  # (n, d)
      difference = permuted - x[None, :]  # (n, d)
      exponent = (difference * difference / (per_coordinate**2)[None, :]).sum(axis=1)
      term = amplitude * np.exp(-0.5 * exponent)  # (n,)
      total_k += term
      # d/dx_j exp(-|x - Xp|^2 / 2 l^2) = term * (Xp_j - x_j) / l_j^2
      total_jac += term[:, None] * difference / (per_coordinate**2)[None, :]
    n_permutations = len(self._permutations)
    return total_k / n_permutations, total_jac / n_permutations

  def diag(self, X):
    """NOT constant: ``k(x, x)`` averages ``exp(-|x - pi.x|^2/2)`` over the group, so it depends on
    how far ``x`` sits from its own permutations (it is ``amplitude`` only for a fully tied design).

    Computed ROW-WISE, in ``O(n |G| d)``. Forming the full Gram and taking its diagonal is
    ``O(n^2 |G| d)``, and ``predict(return_std=True)`` calls this on the entire EI candidate sweep
    (thousands of points), where the difference is seconds per proposal, not microseconds."""
    X = np.atleast_2d(X)
    blocks, free = self._groups
    length_scale = self._length_scales()
    total = np.zeros(X.shape[0])
    for order in self._permutations:
      permuted = self._permuted(X, order)
      scaled = np.zeros(X.shape[0])
      for i, block in enumerate(blocks):
        difference = X[:, block] - permuted[:, block]
        scaled += np.einsum("ij,ij->i", difference, difference) / length_scale[i]**2
      for j, coordinate in enumerate(free):  # a free coordinate is never permuted -> contributes 0
        difference = X[:, coordinate] - permuted[:, coordinate]
        scaled += difference * difference / length_scale[len(blocks) + j]**2
      total += float(self.constant_value) * np.exp(-0.5 * scaled)
    return total / len(self._permutations)

  def grad_diag(self, x):
    """``d k(x, x) / dx`` at a single point ``x`` ``(d,)``.

    Needed because :meth:`diag` is NOT constant for a group-averaged kernel, so the posterior
    variance ``diag(x) - k^T K^-1 k`` has a first term that moves with ``x``. Dropping it makes the
    acquisition's gradient wrong -- not merely imprecise: it can point in the opposite direction,
    and L-BFGS then reports a line-search failure or silently declines to move at all.

    Each permutation contributes ``exp(-E_pi/2)`` with ``E_pi = sum_g |x_g - (pi.x)_g|^2 / l_g^2``,
    whose derivative picks up the permutation BOTH ways round -- ``x_j`` appears in the term as
    itself and as the image of ``pi^-1(j)`` -- hence the two differences below."""
    x = np.asarray(x, dtype=float).ravel()
    blocks, free = self._groups
    length_scale = self._length_scales()
    row = x[None, :]
    gradient = np.zeros(self.d)
    for order in self._permutations:
      permuted = self._permuted(row, order)[0]
      inverse = np.empty_like(np.asarray(order)) if len(order) > 0 else np.asarray(order)
      if len(order) > 0:
        inverse[list(order)] = np.arange(len(order))
      back = self._permuted(row, tuple(inverse))[0] if len(order) > 0 else permuted
      exponent = 0.0
      for i, block in enumerate(blocks):
        difference = x[block] - permuted[block]
        exponent += float(difference @ difference) / length_scale[i]**2
      for j, coordinate in enumerate(free):
        difference = x[coordinate] - permuted[coordinate]
        exponent += float(difference * difference) / length_scale[len(blocks) + j]**2
      weight = float(self.constant_value) * np.exp(-0.5 * exponent)
      for i, block in enumerate(blocks):
        gradient[block] -= weight * ((x[block] - permuted[block]) + (x[block] - back[block])) / length_scale[i]**2
    return gradient / len(self._permutations)

  def is_stationary(self):
    return len(self._permutations) == 1  # only the degenerate (no-symmetry) case is stationary

  def __repr__(self):
    blocks, free = self._groups
    return (
      f"{type(self).__name__}(d={self.d}, blocks={len(blocks)}x{blocks[0].size if blocks else 0}, "
      f"free={free.size}, amplitude^2={float(self.constant_value):.3g}, "
      f"l={np.round(self._length_scales(), 3)})"
    )


class SortingRBF(ARDRBF):
  """An ARD-RBF that SORTS the exchangeable elements before comparing them.

  ``k(x, y) = k_ard(sigma x, sigma y)``, where ``sigma`` orders the elements by one designated block
  (the sort KEY) and applies that same order to every co-permuted block -- for the enzyme batch,
  ordering by temperature carries each experiment's fraction with it, because experiment ``k`` is the
  pair.

  Invariance by FOLDING rather than by averaging, and the trade against
  :class:`PermutationInvariantRBF` is worth stating plainly:

  * **Exactly invariant and PSD for free.** ``sigma(pi x) = sigma(x)``, and composing any map with a
    PSD kernel preserves positive-definiteness.
  * **Constant diagonal.** ``k(x, x) = amplitude`` everywhere, because both arguments sort to the
    same point. The group average instead peaks at ``|G|`` times the generic value on the fully-tied
    stratum -- correct for that construction (it is the method of images at a reflecting boundary),
    but it makes EI's exploration term maximal exactly on the degenerate all-identical batch, which
    is the least informative design of experiments there is.
  * **ARD acts on ORDER STATISTICS.** After sorting, coordinate ``k`` is the k-th smallest, which a
    symmetric function *can* distinguish -- unlike experiment slots, which it cannot. Measured on the
    enzyme benchmark the order statistics carry real signal (Spearman -0.22, -0.19, +0.08, +0.32
    against the loss) where the raw slots carry none (-0.04 .. +0.01).
  * **For scalar keys it realises the exact quotient metric**: ``||sort a - sort b|| = min_pi ||a - pi b||``
    (the rearrangement inequality), so lengthscales keep their physical meaning.
  * **The price is a crease at ties.** The one-sided derivatives across ``x_i = x_j`` differ by
    ``c E (w_j - w_i) / l^2`` -- driven by the spread of the OTHER point, so a shared lengthscale
    does not remove it. Deliberately accepted: the tie locus has measure zero, L-BFGS-B tolerates a
    kink, and the group average's smoothness costs the pathology above.

  Measured against the alternatives on the shared-fraction enzyme design (5 BO seeds, 30 iterations,
  scripts/benchmark_symmetry.py): sorting 0.0424 and the only surrogate to beat the random null
  (1.29x, p = 0.016); plain ARD 0.0555 and a stick-breaking reparameterisation 0.0559, both
  indistinguishable from the null (p = 0.42 and 0.79).
  """

  def __init__(
    self, d, sort_blocks=(), key=-1, constant_value=1.0, constant_value_bounds=(1e-6, 1e2), length_scale=1.0,
    length_scale_bounds=(1e-2, 1e1)
  ):
    # The parent is the PLAIN ARD-RBF: this kernel is a per-coordinate ARD in the sorted frame, so
    # every coordinate keeps its own lengthscale, and the invariance comes from the folding below
    # rather than from any group the parent averages over. `sort_blocks` describes only which
    # coordinates travel together under the sort.
    super().__init__(
      d=d, constant_value=constant_value, constant_value_bounds=constant_value_bounds, length_scale=length_scale,
      length_scale_bounds=length_scale_bounds
    )
    self.sort_blocks = sort_blocks
    self.key = key
    # Every name in the signature must survive `get_params` -> `clone`, which sklearn calls on each
    # fit; a `**kwargs` signature silently yields an empty `theta` and an unfittable kernel.
    self._sort_index = np.asarray([np.asarray(b, dtype=int) for b in sort_blocks], dtype=int) \
        if len(sort_blocks) > 0 else np.empty((0, 0), dtype=int)

  def _order(self, X):
    """The permutation that sorts each row by the key block."""
    return np.argsort(X[:, self._sort_index[self.key]], axis=1)

  def _sorted(self, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if self._sort_index.size == 0:
      return X
    out = np.array(X, copy=True)
    order = self._order(X)
    for block in self._sort_index:
      out[:, block] = np.take_along_axis(X[:, block], order, axis=1)
    return out

  def __call__(self, X, Y=None, eval_gradient=False):
    return super().__call__(self._sorted(X), None if Y is None else self._sorted(Y), eval_gradient)

  def diag(self, X):
    return super().diag(self._sorted(X))

  # `grad_diag` is the parent's: sorting leaves k(x, x) at the amplitude, so the zero it returns is
  # already the right answer here -- and the reason this kernel exists is that the group average's
  # is NOT zero, which is what pulls EI toward self-similar designs.

  def k_and_grad_x(self, X_train, x):
    """Chain rule through the sort. Its Jacobian is a permutation matrix almost everywhere, so the
    gradient computed in the sorted frame is scattered back to the coordinates it came from.
    Undefined on the tie locus itself (measure zero), where the two one-sided values differ."""
    x = np.asarray(x, dtype=float).ravel()
    k, jac = super().k_and_grad_x(self._sorted(X_train), self._sorted(x))
    if self._sort_index.size == 0:
      return k, jac
    order = self._order(x[None, :])[0]
    scattered = np.array(jac, copy=True)
    for block in self._sort_index:
      scattered[:, block[order]] = jac[:, block]
    return k, scattered

  def is_stationary(self):
    return self._sort_index.size == 0

  def __repr__(self):
    return (
      f"{type(self).__name__}(d={self.d}, sort_blocks={len(self.sort_blocks)}x"
      f"{self._sort_index.shape[1] if self._sort_index.size else 0}, key={self.key}, "
      f"amplitude^2={float(self.constant_value):.3g}, l={np.round(self.coordinate_length_scales(), 3)})"
    )


class IndependentSortingRBF(ARDRBF):
  """An ARD-RBF that sorts each block by ITS OWN values, independently of every other block.

  ``k(x, y) = k_ard(sigma x, sigma y)``, where ``sigma`` sorts block ``b`` into ascending order using
  block ``b``'s own entries. The invariance realised is the PRODUCT group ``S_m x ... x S_m``, one
  factor per block, rather than the single diagonal ``S_m`` of :class:`SortingRBF`.

  That is the difference between the two, and it is the whole reason this class exists.
  :class:`SortingRBF` orders by one designated KEY block and carries every other block along in that
  same order, because for the enzyme batch experiment ``k`` IS the pair ``(fraction_k, temperature_k)``
  and separating them would scramble the experiments. Here the blocks are not tied: the MNIST window
  names two opposite corners, ``(x1, x2)`` and ``(y1, y2)``, and swapping ``x1`` with ``x2`` names the
  same window whether or not ``y1`` and ``y2`` are also swapped. Sorting each pair on its own maps
  every one of the 4 equivalent designs to the canonical ``(left, right, top, bottom)``, which is
  exactly the surrogate the objective deserves: the loss is a function of the window, and a quarter
  of the unit cube already contains every window there is.

  It inherits SortingRBF's properties, for the same reasons:

  * **Exactly invariant and PSD for free**, since composing a map with a PSD kernel preserves
    positive-definiteness, and sorting is idempotent under the group.
  * **Constant diagonal**, ``k(x, x) = amplitude``, so EI's exploration term is not inflated on the
    tied stratum.
  * **ARD acts on ORDER STATISTICS.** After sorting, the four coordinates are ``left``, ``right``,
    ``top`` and ``bottom``, each with its own lengthscale -- and unlike the raw corners those are
    quantities the loss is genuinely a function of.
  * **A crease on the tie locus** (``x1 = x2``, the degenerate zero-width window), which has measure
    zero and which L-BFGS-B tolerates.

  ``sort_blocks`` lists the flat design indices of each block. They need not be the same length."""

  def __init__(
    self, d, sort_blocks=(), constant_value=1.0, constant_value_bounds=(1e-6, 1e2), length_scale=1.0,
    length_scale_bounds=(1e-2, 1e1)
  ):
    # As in SortingRBF, the parent is the PLAIN ARD-RBF -- a per-coordinate ARD in the sorted frame.
    # There is no `key` here: that argument names the block whose order the others follow, and the
    # absence of such a block is precisely what distinguishes this kernel.
    super().__init__(
      d=d, constant_value=constant_value, constant_value_bounds=constant_value_bounds, length_scale=length_scale,
      length_scale_bounds=length_scale_bounds
    )
    self.sort_blocks = sort_blocks
    # Every name in the signature must survive `get_params` -> `clone`, which sklearn calls on each
    # fit; a `**kwargs` signature silently yields an empty `theta` and an unfittable kernel.
    self._blocks = tuple(np.asarray(b, dtype=int) for b in sort_blocks)

  def _sorted(self, X):
    """Each block ascending, in place, on a copy."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if len(self._blocks) == 0:
      return X
    out = np.array(X, copy=True)
    for block in self._blocks:
      out[:, block] = np.sort(X[:, block], axis=1)
    return out

  def __call__(self, X, Y=None, eval_gradient=False):
    return super().__call__(self._sorted(X), None if Y is None else self._sorted(Y), eval_gradient)

  def diag(self, X):
    return super().diag(self._sorted(X))

  # `grad_diag` is the parent's, and its zero is right here: sorting leaves k(x, x) at the amplitude.

  def k_and_grad_x(self, X_train, x):
    """Chain rule through the sort. Its Jacobian is a permutation matrix almost everywhere -- a
    BLOCK-DIAGONAL one, each block permuted by its own order -- so the gradient computed in the
    sorted frame is scattered back to the coordinates it came from. Undefined on the tie locus
    itself (measure zero), where the two one-sided values differ."""
    x = np.asarray(x, dtype=float).ravel()
    k, jac = super().k_and_grad_x(self._sorted(X_train), self._sorted(x))
    if len(self._blocks) == 0:
      return k, jac
    scattered = np.array(jac, copy=True)
    for block in self._blocks:
      order = np.argsort(x[block])
      scattered[:, block[order]] = jac[:, block]
    return k, scattered

  def is_stationary(self):
    return len(self._blocks) == 0

  def __repr__(self):
    return (
      f"{type(self).__name__}(d={self.d}, sort_blocks={[b.size for b in self._blocks]}, "
      f"amplitude^2={float(self.constant_value):.3g}, l={np.round(self.coordinate_length_scales(), 3)})"
    )


class NormalisedInvariantRBF(PermutationInvariantRBF):
  """The group average, rescaled to a CONSTANT diagonal: ``amplitude^2 * C(x, y)`` with the
  correlation ``C(x, y) = P(x, y) / sqrt(P(x, x) P(y, y))``.

  Keeps what makes the group average principled -- exact invariance, and smoothness across the tie
  locus, which :class:`SortingRBF` gives up -- while removing the property that misdirects the
  acquisition. The unnormalised average's prior variance is up to ``|G|`` times larger on the fully
  tied stratum than at a generic design, so EI's exploration term peaks exactly on the all-identical
  batch, the least informative design of experiments there is. Measured as the ratio of mean
  ``diag`` over 2000 tied against 2000 generic designs, enzyme layout, ``|G| = 24``: **6.3x at the
  config's initial lengthscale** (``exp(-1) = 0.368``, the midpoint of the prior bounds), 4.5x at
  ``l = 0.45``, and 23.0x at ``l = 0.05`` -- i.e. rising toward ``|G|`` as the lengthscale shortens.

  Still PSD: this is ``D^{-1/2} K D^{-1/2}`` scaled by a positive constant, a congruence. Still
  exactly invariant, since ``K`` and its diagonal both are. What it gives up is the interpretation --
  the group average's varying diagonal IS the method-of-images (Neumann) prior on the fundamental
  domain, so normalising it means no longer modelling that.

  THE AMPLITUDE IS KEPT OUTSIDE THE NORMALISATION, and that is not a detail. Normalising as the bare
  ``K / sqrt(k(x,x) k(y,y))`` cancels ``constant_value`` identically -- both numerator and
  denominator carry one factor of it -- leaving ``k(x, x) = 1`` for every amplitude and
  ``dk / dlog(constant_value) == 0`` exactly. The marginal likelihood is then FLAT in the amplitude,
  the GP cannot scale its prior to the data, and with the enzyme losses (variance ~1e-3 .. 1e-2
  after centring) a prior variance pinned at 1 inflates the posterior everywhere, so EI over-explores
  and the surrogate degenerates toward random search. That is how `scripts/benchmark_symmetry.py`
  implements it, and it is the most likely reason the normalised kernel came LAST there (0.86x, behind
  the random null) -- a defect of that implementation, not a property of normalisation.
  """

  def __call__(self, X, Y=None, eval_gradient=False):
    amplitude = float(self.constant_value)
    if not eval_gradient:
      correlation = super().__call__(X, Y, False)
      a = super().diag(np.atleast_2d(X))
      b = a if Y is None else super().diag(np.atleast_2d(Y))
      return amplitude * correlation / np.sqrt(np.outer(a, b))
    # The gradient is only defined for the symmetric call, so a == b below. `Y` is FORWARDED rather
    # than replaced by None: the base raises for a cross-gradient request, and hard-coding None here
    # would swallow that check and silently return k(X, X) and its gradient whenever the caller
    # passed a Y of the same length. sklearn's GPR never asks, but its Sum/Product wrappers do.
    K, dK = super().__call__(X, Y, True)
    a = np.diag(K).copy()
    a_grad = np.einsum("iit->it", dK)  # d diag / d theta, (n, n_theta)
    norm = np.sqrt(np.outer(a, a))
    correlation = K / norm
    # d(K/N)/dt = dK/dt / N - (K/N) * ( a'_i/(2 a_i) + a'_j/(2 a_j) )
    share = a_grad / (2.0 * a[:, None])
    d_correlation = dK / norm[:, :, None] - correlation[:, :, None] * (share[:, None, :] + share[None, :, :])
    value, gradient = amplitude * correlation, amplitude * d_correlation
    if not self.hyperparameter_constant_value.fixed:
      # The correlation is amplitude-free, so its own log-amplitude column is identically zero; the
      # value is linear in the amplitude, so this column is the value itself.
      gradient[:, :, 0] = value
    return value, gradient

  def diag(self, X):
    """CONSTANT, unlike the group average this normalises: ``k(x, x) = amplitude^2`` everywhere."""
    return np.full(np.atleast_2d(X).shape[0], float(self.constant_value))

  def grad_diag(self, x):
    return np.zeros(self.d)

  def k_and_grad_x(self, X_train, x):
    x = np.asarray(x, dtype=float).ravel()
    k, jac = super().k_and_grad_x(X_train, x)
    kxx = float(super().diag(x[None, :])[0])
    a = super().diag(np.atleast_2d(X_train))
    norm = np.sqrt(kxx * a)
    correlation = k / norm
    # Only k(x, x) depends on x, so the quotient rule leaves a single correction term.
    d_correlation = jac / norm[:, None] - correlation[:, None] * super().grad_diag(x)[None, :] / (2.0 * kxx)
    amplitude = float(self.constant_value)
    return amplitude * correlation, amplitude * d_correlation

  def __repr__(self):
    # Per-GROUP lengthscales, as the parent prints them -- NOT `coordinate_length_scales()`, which
    # repeats each block's value across its coordinates and would show 8 numbers for a kernel that
    # has 2. This kernel's whole point is that a symmetric function cannot tell experiment 1 from
    # experiment 3, so a repr suggesting 8 free lengthscales asserts the opposite of what it is.
    blocks, free = self._groups
    return (
      f"{type(self).__name__}(d={self.d}, blocks={len(blocks)}x"
      f"{blocks[0].size if len(blocks) > 0 else 0}, free={free.size}, "
      f"amplitude^2={float(self.constant_value):.3g}, l={np.round(self._length_scales(), 3)})"
    )
