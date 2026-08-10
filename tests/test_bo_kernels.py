"""The permutation-invariant GP kernel.

Everything here is checked against a numeric reference rather than asserted, because two of the
three pieces (the hyperparameter gradient sklearn's `fit` optimises, and the input gradient the EI
polish uses) are silent when wrong: a bad gradient gives a worse fit and a worse proposal, not an
error.
"""

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from detopt.bo import PermutationInvariantRBF, SortingRBF, __kernels__, kernel_from_config

BLOCKS = ((0, 1, 2, 3), (4, 5, 6, 7))  # the enzyme layout: fraction block + temperature block


def _kernel(blocks=BLOCKS, d=8, length_scale=None):
  n_groups = (len(blocks) + (d - sum(len(b) for b in blocks))) if blocks else d
  return PermutationInvariantRBF(
    d=d, blocks=blocks, constant_value=1.3, constant_value_bounds=(1e-3, 1e3),
    length_scale=np.full(n_groups, 0.45) if length_scale is None else length_scale,
    length_scale_bounds=(1e-2, 1e2)
  )


def _permute(X, order, blocks=BLOCKS):
  out = np.array(X, copy=True)
  for block in blocks:
    out[:, list(block)] = X[:, np.asarray(block)[list(order)]]
  return out


def test_registry_names_and_config_entry():
  """Four named modelling CHOICES: no symmetry, symmetry by group average, that average rescaled to
  a constant diagonal, and symmetry by sorting."""
  assert set(__kernels__) == {"ard-rbf", "permutation-invariant-rbf", "normalised-invariant-rbf",
                              "sorting-rbf"}


def test_lengthscale_count_is_per_group_not_per_coordinate():
  """The whole point: an exchangeable block shares ONE lengthscale, and blocks that are never
  permuted into each other keep their own. 8 design coordinates -> 2 lengthscales."""
  assert _kernel().n_length_scales == 2
  assert _kernel(blocks=()).n_length_scales == 8  # no symmetry -> ordinary ARD
  # a free coordinate keeps its own
  assert _kernel(blocks=((0, 1, 2, 3),), d=6, length_scale=np.full(3, 0.4)).n_length_scales == 3


def test_kernel_is_invariant_under_permuting_either_argument():
  rng = np.random.default_rng(0)
  X, Y = rng.random((5, 8)), rng.random((7, 8))
  kernel = _kernel()
  base = kernel(X, Y)
  for order in ((3, 2, 1, 0), (1, 3, 0, 2)):
    np.testing.assert_allclose(kernel(_permute(X, order), Y), base, atol=1e-12)
    np.testing.assert_allclose(kernel(X, _permute(Y, order)), base, atol=1e-12)


def test_gram_matrix_is_symmetric_and_positive_semidefinite():
  rng = np.random.default_rng(1)
  X = rng.random((24, 8))
  K = _kernel()(X)
  np.testing.assert_allclose(K, K.T, atol=1e-12)
  assert np.min(np.linalg.eigvalsh(K)) > -1e-9


def test_diag_matches_the_gram_diagonal_and_is_not_constant():
  """`diag` is overridden because a group-averaged kernel's k(x, x) depends on how far x sits from
  its own permutations -- it is the amplitude only for a fully tied design."""
  rng = np.random.default_rng(2)
  X = rng.random((10, 8))
  kernel = _kernel()
  np.testing.assert_allclose(kernel.diag(X), np.diag(kernel(X)), atol=1e-12)
  assert np.ptp(kernel.diag(X)) > 1e-3, "diag looks constant; the symmetry is not being applied"
  tied = np.tile(rng.random((1, 4)), 2).reshape(1, 8)  # every experiment identical -> all orders equal
  tied = np.concatenate([np.full((1, 4), 0.3), np.full((1, 4), 0.7)], axis=1)
  np.testing.assert_allclose(kernel.diag(tied), float(kernel.constant_value), atol=1e-12)


def test_hyperparameter_gradient_matches_finite_differences():
  """This is what sklearn's `fit` maximises the marginal likelihood with."""
  rng = np.random.default_rng(3)
  X = rng.random((9, 8))
  kernel = _kernel()
  _, gradient = kernel(X, eval_gradient=True)

  theta0 = kernel.theta.copy()
  step = 1e-6
  for k in range(theta0.size):
    up, down = theta0.copy(), theta0.copy()
    up[k] += step
    down[k] -= step
    kernel.theta = up
    K_up = kernel(X)
    kernel.theta = down
    K_down = kernel(X)
    kernel.theta = theta0
    np.testing.assert_allclose(gradient[:, :, k], (K_up - K_down) / (2 * step), rtol=1e-4, atol=1e-6)


def test_input_gradient_matches_finite_differences():
  """`k_and_grad_x` feeds the analytic EI gradient. For a group-averaged kernel it is the average of
  the per-permutation gradients, NOT the ARD-RBF expression at some effective distance."""
  rng = np.random.default_rng(4)
  X_train = rng.random((11, 8))
  x = rng.random(8)
  kernel = _kernel()

  k, jac = kernel.k_and_grad_x(X_train, x)
  np.testing.assert_allclose(k, kernel(x[None, :], X_train)[0], atol=1e-12)

  step = 1e-6
  for j in range(8):
    up, down = x.copy(), x.copy()
    up[j] += step
    down[j] -= step
    numeric = (kernel(up[None, :], X_train)[0] - kernel(down[None, :], X_train)[0]) / (2 * step)
    np.testing.assert_allclose(jac[:, j], numeric, rtol=1e-4, atol=1e-7)


def test_without_blocks_it_is_exactly_an_ard_rbf():
  """The SST path. No exchangeable block -> the group is trivial and the kernel must reproduce
  sklearn's own ConstantKernel * RBF to machine precision, so nothing changes for those detectors."""
  rng = np.random.default_rng(5)
  X, Y = rng.random((6, 5)), rng.random((4, 5))
  length_scale = np.array([0.2, 0.5, 1.0, 0.3, 0.8])
  mine = PermutationInvariantRBF(d=5, blocks=(), constant_value=2.0, length_scale=length_scale)
  reference = ConstantKernel(2.0) * RBF(length_scale)
  np.testing.assert_allclose(mine(X, Y), reference(X, Y), atol=1e-12)
  np.testing.assert_allclose(mine(X), reference(X), atol=1e-12)
  np.testing.assert_allclose(mine.diag(X), reference.diag(X), atol=1e-12)


def test_sklearn_fits_it_and_predictions_are_invariant():
  """End to end: hyperparameter fitting runs, and the fitted GP's PREDICTION is invariant."""
  rng = np.random.default_rng(6)
  X = rng.random((40, 8))
  y = np.array([np.sum(np.sort(row[4:])[-2:]) + 0.1 * np.sum(row[:4]) for row in X])  # symmetric

  model = GaussianProcessRegressor(kernel=_kernel(), alpha=1e-8, normalize_y=False).fit(X, y)
  assert np.isfinite(model.log_marginal_likelihood_value_)

  probe = rng.random((1, 8))
  for order in ((2, 0, 3, 1), (3, 2, 1, 0)):
    np.testing.assert_allclose(model.predict(_permute(probe, order)), model.predict(probe), atol=1e-8)


def test_config_factory_builds_from_the_detector():
  import detopt
  from detopt.utils.config import load_config

  detector = detopt.detector.from_config(load_config("config/detector/enzyme.yaml"))
  gp = {"log_lengthscale_prior_bounds": [-6.0, 4.0], "log_amplitude_prior_bounds": [-6.0, 1.5]}

  kernel = kernel_from_config({"permutation-invariant-rbf": {"exchangeable": detector.n_experiments}}, detector, gp)
  assert kernel.d == detector.design_dim()
  assert kernel.n_length_scales == 2  # fraction block + temperature block

  plain = kernel_from_config({"ard-rbf": {}}, detector, gp)
  assert plain.n_length_scales == detector.design_dim()

  with pytest.raises(ValueError, match="needs .exchangeable"):
    kernel_from_config({"permutation-invariant-rbf": {}}, detector, gp)
  with pytest.raises(ValueError, match="models no symmetry"):
    kernel_from_config({"ard-rbf": {"exchangeable": 4}}, detector, gp)


def test_ryser_permanent_matches_the_definition():
  """The permanent IS the sum over permutations of the product of matched entries -- the determinant
  without the alternating sign. Ryser computes it in O(2^m m) instead of O(m! m); this pins the two
  against each other, because the kernel VALUE goes through Ryser while its hyperparameter gradient
  goes through the explicit permutation sum, and a mismatch would silently corrupt the fit."""
  import itertools

  rng = np.random.default_rng(7)
  for m in (1, 2, 3, 4, 5):
    A = rng.random((3, m, m))
    brute = np.array([
      sum(np.prod([a[k, order[k]] for k in range(m)]) for order in itertools.permutations(range(m)))
      for a in A
    ])
    np.testing.assert_allclose(PermutationInvariantRBF._permanent(A), brute, rtol=1e-10)

  # a 0/1 matrix's permanent counts perfect matchings: the all-ones 4x4 has 4! = 24
  np.testing.assert_allclose(PermutationInvariantRBF._permanent(np.ones((1, 4, 4))), [24.0], rtol=1e-10)
  np.testing.assert_allclose(PermutationInvariantRBF._permanent(np.eye(4)[None]), [1.0], atol=1e-10)


# --------------------------------------------------------------------------- #
# Independent JAX reference + autodiff.
#
# The numpy kernel is fast but clever: the value goes through Ryser's formula and BOTH gradients are
# analytic. Finite differences catch gross errors; they do not catch a wrong-by-a-constant gradient
# hidden inside the same code path. So the reference below is a deliberately NAIVE re-implementation
# -- brute force over permutations, direct distances, no permanent -- differentiated by autodiff.
# Two independent implementations and two independent differentiation methods.
# --------------------------------------------------------------------------- #
def _jax_kernel(theta, X, Y, blocks, free, n_blocks):
  """k(X, Y) rebuilt in JAX from sklearn's `theta` (= log of [constant_value, length_scale...])."""
  import itertools

  import jax.numpy as jnp

  amplitude = jnp.exp(theta[0])
  length_scale = jnp.exp(theta[1:])

  factor = jnp.ones((X.shape[0], Y.shape[0]))
  for j, coordinate in enumerate(free):
    difference = X[:, coordinate][:, None] - Y[:, coordinate][None, :]
    factor = factor * jnp.exp(-0.5 * difference**2 / length_scale[n_blocks + j] ** 2)
  if len(blocks) == 0:
    return amplitude * factor

  m = len(blocks[0])
  total = jnp.zeros((X.shape[0], Y.shape[0]))
  for order in itertools.permutations(range(m)):
    exponent = jnp.zeros((X.shape[0], Y.shape[0]))
    for b, block in enumerate(blocks):
      permuted = [block[k] for k in order]
      difference = X[:, list(block)][:, None, :] - Y[:, permuted][None, :, :]
      exponent = exponent + jnp.sum(difference**2, axis=-1) / length_scale[b] ** 2
    total = total + jnp.exp(-0.5 * exponent)
  return amplitude * factor * total / float(np.prod(np.arange(1, m + 1)))



@pytest.fixture(autouse=True)
def _x64():
  """``jax_enable_x64`` is a GLOBAL flag with no context manager in this jax version, and four tests
  below need it to compare float64 numpy against JAX. Leaking it breaks ``detopt.bo.jax_gp``, whose
  float32 literals make ``lax.cond`` branches disagree on dtype -- so the whole suite passes today
  only because pytest collects ``test_bo_gp`` before ``test_bo_kernels``. Restore it here, as
  ``tests/test_bo_ei_gradient.py`` already does, so no ordering can turn those tests red."""
  jax = pytest.importorskip("jax")
  previous = jax.config.jax_enable_x64
  yield
  jax.config.update("jax_enable_x64", previous)

def test_value_matches_an_independent_jax_implementation():
  jax = pytest.importorskip("jax")
  jnp = pytest.importorskip("jax.numpy")
  jax.config.update("jax_enable_x64", True)  # else this compares float64 numpy against float32 JAX
  rng = np.random.default_rng(10)
  X, Y = rng.random((6, 8)), rng.random((5, 8))
  kernel = _kernel()

  reference = _jax_kernel(jnp.asarray(kernel.theta, dtype=jnp.float64), jnp.asarray(X, dtype=jnp.float64),
                          jnp.asarray(Y, dtype=jnp.float64), [list(b) for b in BLOCKS], [], len(BLOCKS))
  np.testing.assert_allclose(kernel(X, Y), np.asarray(reference), rtol=1e-10, atol=1e-12)


def test_hyperparameter_gradient_matches_autodiff():
  """sklearn's `fit` optimises the marginal likelihood with this gradient; a constant factor wrong
  here degrades every fit silently."""
  jax = pytest.importorskip("jax")
  jnp = pytest.importorskip("jax.numpy")
  jax.config.update("jax_enable_x64", True)

  rng = np.random.default_rng(11)
  X = rng.random((7, 8))
  kernel = _kernel()
  _, analytic = kernel(X, eval_gradient=True)

  blocks = [list(b) for b in BLOCKS]
  f = lambda theta: _jax_kernel(theta, jnp.asarray(X), jnp.asarray(X), blocks, [], len(BLOCKS))
  autodiff = np.asarray(jax.jacrev(f)(jnp.asarray(kernel.theta, dtype=jnp.float64)))
  np.testing.assert_allclose(analytic, autodiff, rtol=1e-8, atol=1e-10)


def test_input_gradient_matches_autodiff():
  """`k_and_grad_x` feeds the analytic EI gradient inside the L-BFGS polish."""
  jax = pytest.importorskip("jax")
  jnp = pytest.importorskip("jax.numpy")
  jax.config.update("jax_enable_x64", True)

  rng = np.random.default_rng(12)
  X_train = rng.random((9, 8))
  x = rng.random(8)
  kernel = _kernel()
  value, analytic = kernel.k_and_grad_x(X_train, x)

  blocks = [list(b) for b in BLOCKS]
  theta = jnp.asarray(kernel.theta, dtype=jnp.float64)
  f = lambda z: _jax_kernel(theta, z[None, :], jnp.asarray(X_train), blocks, [], len(BLOCKS))[0]
  np.testing.assert_allclose(value, np.asarray(f(jnp.asarray(x))), rtol=1e-10)
  autodiff = np.asarray(jax.jacrev(f)(jnp.asarray(x, dtype=jnp.float64)))  # (n, d)
  np.testing.assert_allclose(analytic, autodiff, rtol=1e-8, atol=1e-10)


def test_free_coordinates_also_match_autodiff():
  """A design with a coordinate OUTSIDE every exchangeable block: it keeps its own lengthscale and
  factorises out of the permanent, which is a separate branch in both the value and the gradient."""
  jax = pytest.importorskip("jax")
  jnp = pytest.importorskip("jax.numpy")
  jax.config.update("jax_enable_x64", True)

  blocks, free, d = [(0, 1, 2), (3, 4, 5)], [6, 7], 8
  kernel = PermutationInvariantRBF(
    d=d, blocks=tuple(tuple(b) for b in blocks), constant_value=0.9,
    constant_value_bounds=(1e-3, 1e3), length_scale=np.array([0.4, 0.7, 0.3, 1.1]),
    length_scale_bounds=(1e-2, 1e2)
  )
  assert kernel.n_length_scales == 4  # 2 blocks + 2 free coordinates

  rng = np.random.default_rng(13)
  X = rng.random((6, d))
  _, analytic = kernel(X, eval_gradient=True)
  f = lambda theta: _jax_kernel(theta, jnp.asarray(X), jnp.asarray(X), blocks, free, len(blocks))
  autodiff = np.asarray(jax.jacrev(f)(jnp.asarray(kernel.theta, dtype=jnp.float64)))
  np.testing.assert_allclose(analytic, autodiff, rtol=1e-8, atol=1e-10)


def test_grad_diag_matches_finite_differences():
  """``diag`` is NOT constant for a group-averaged kernel, so the EI gradient needs its derivative.
  That term had no test at all, and it is the one the acquisition's polish is most sensitive to."""
  rng = np.random.default_rng(4)
  for blocks, d in ((((0, 1, 2, 3), (4, 5, 6, 7)), 8), (((1, 2, 3, 4),), 5), (((0, 1), (2, 3)), 5), ((), 4)):
    kernel = PermutationInvariantRBF(d=d, blocks=blocks, constant_value=1.3, length_scale=0.6)
    x = rng.random(d)
    analytic = kernel.grad_diag(x)
    numeric = np.zeros(d)
    for j in range(d):
      step = np.zeros(d)
      step[j] = 1e-6
      numeric[j] = (kernel.diag((x + step)[None, :])[0] - kernel.diag((x - step)[None, :])[0]) / 2e-6
    assert np.allclose(analytic, numeric, atol=1e-7), (blocks, analytic, numeric)


def test_grad_diag_is_exactly_zero_without_a_symmetry():
  """With no exchangeable block the kernel is stationary and ``diag`` is the amplitude, so the term
  must vanish EXACTLY -- the ARD path has to stay bit-identical to the stationary formula."""
  kernel = PermutationInvariantRBF(d=6, blocks=(), constant_value=2.0, length_scale=0.4)
  assert np.all(kernel.grad_diag(np.random.default_rng(0).random(6)) == 0.0)


def test_prior_variance_is_maximal_on_the_tied_diagonal():
  """A group average is largest where the design is FIXED by the group: k(x,x) = amplitude on the
  fully-tied stratum against ~amplitude/|G| at a generic design.

  This is a property of the kernel, not a bug in it -- but it is the mechanism by which EI's
  exploration term prefers a batch whose experiments are all identical, i.e. the least informative
  design of experiments there is. Pinned here so that a future normalisation (k/sqrt(k(x,x)k(y,y)))
  is a deliberate change with a failing test to update, rather than a silent one."""
  m = 4
  kernel = PermutationInvariantRBF(d=2 * m, blocks=(tuple(range(m)), tuple(range(m, 2 * m))),
                                   constant_value=1.0, length_scale=0.37)
  rng = np.random.default_rng(1)
  tied = np.repeat(rng.random(2), m).reshape(1, -1)
  generic = rng.random((1, 2 * m))
  assert np.isclose(kernel.diag(tied)[0], 1.0)
  assert kernel.diag(tied)[0] > 5.0 * kernel.diag(generic)[0]


def _sorting(d=8, m=4, **kwargs):
  return SortingRBF(d=d, sort_blocks=(tuple(range(m)), tuple(range(m, 2 * m))), key=-1,
                    constant_value=1.3, length_scale=np.array([0.3, 0.5, 0.4, 0.6, 0.35, 0.45, 0.55, 0.25]),
                    **kwargs)


def test_sorting_kernel_is_exactly_invariant():
  """Relabelling the batch -- fractions carried along with their own temperatures -- must not move
  the kernel by a single ulp, for every one of the m! orderings."""
  import itertools

  kernel, rng = _sorting(), np.random.default_rng(0)
  X, x = rng.random((5, 8)), rng.random(8)
  values = []
  for order in itertools.permutations(range(4)):
    permuted = np.concatenate([x[:4][list(order)], x[4:][list(order)]])
    values.append(kernel(permuted[None, :], X))
  assert np.ptp(np.stack(values), axis=0).max() == 0.0


def test_sorting_kernel_has_a_constant_diagonal():
  """The whole reason to prefer this over the group average: k(x, x) does NOT peak on the
  all-identical batch, so EI's exploration term has no pull toward the least informative design."""
  kernel = _sorting()
  tied = np.concatenate([np.full(4, 0.4), np.full(4, 0.7)])[None, :]
  spread = np.array([[0.1, 0.4, 0.6, 0.9, 0.15, 0.35, 0.75, 0.95]])
  assert np.isclose(kernel.diag(tied)[0], 1.3)
  assert np.isclose(kernel.diag(spread)[0], 1.3)
  assert np.all(kernel.grad_diag(spread[0]) == 0.0)


def test_sorting_kernel_input_gradient_matches_finite_differences():
  """``k_and_grad_x`` must scatter the sorted-frame gradient back through the INVERSE permutation;
  getting that wrong leaves the magnitudes right and the directions permuted, which an optimiser
  cannot detect. Checked away from the tie locus, where the kernel is differentiable."""
  kernel, rng = _sorting(), np.random.default_rng(2)
  X = rng.random((6, 8))
  x = np.array([0.20, 0.50, 0.80, 0.35, 0.10, 0.40, 0.90, 0.60])  # all temperatures distinct
  _, jac = kernel.k_and_grad_x(X, x)
  numeric = np.zeros_like(jac)
  for j in range(8):
    step = np.zeros(8)
    step[j] = 1e-6
    numeric[:, j] = (kernel((x + step)[None, :], X)[0] - kernel((x - step)[None, :], X)[0]) / 2e-6
  assert np.allclose(jac, numeric, atol=1e-7)


def test_sorting_kernel_survives_sklearn_clone():
  """sklearn re-clones the kernel on every fit from ``get_params``, so every constructor argument
  must be a named parameter -- a ``**kwargs`` signature yields an empty ``theta`` and a kernel that
  cannot be fitted, silently."""
  from sklearn.base import clone

  kernel = _sorting()
  copy = clone(kernel)
  assert np.allclose(copy.theta, kernel.theta) and copy.bounds.shape == kernel.bounds.shape
  assert kernel.n_length_scales == 8  # one per coordinate: ARD acts on ORDER STATISTICS


# --------------------------------------------------------------------------- #
# The normalised group average: c * k(x,y) / sqrt(k(x,x) k(y,y)).
#
# Written after the SCRIPT-LOCAL version of this kernel (scripts/benchmark_symmetry.py) was found to
# cancel its own amplitude and was benchmarked in that state, coming last. Every test below would
# have failed against that implementation.
# --------------------------------------------------------------------------- #


def _normalised(blocks=BLOCKS, d=8, constant_value=1.3):
  from detopt.bo import NormalisedInvariantRBF

  return NormalisedInvariantRBF(
    d=d, blocks=blocks, constant_value=constant_value, constant_value_bounds=(1e-3, 1e3),
    length_scale=np.full(len(blocks), 0.45), length_scale_bounds=(1e-2, 1e2)
  )


def test_normalised_amplitude_is_identifiable():
  """THE REGRESSION TEST. Normalising as the bare ``k / sqrt(k(x,x) k(y,y))`` cancels the amplitude
  -- numerator and denominator each carry one factor -- leaving ``k(x,x) = 1`` for every value of it
  and a marginal likelihood that is FLAT in that hyperparameter. Nothing raises; the GP simply
  cannot scale its prior to the data, over-explores, and degenerates toward random search."""
  rng = np.random.default_rng(0)
  X = rng.random((5, 8))
  for amplitude in (0.01, 1.0, 100.0):
    kernel = _normalised(constant_value=amplitude)
    K, gradient = kernel(X, None, True)
    assert np.isclose(K[0, 0], amplitude), "k(x, x) must BE the amplitude, not 1"
    assert np.abs(gradient[:, :, 0]).max() > 0.0, "d k / d log(amplitude) is identically zero"
    assert np.allclose(gradient[:, :, 0], K), "k is linear in the amplitude, so this column is k"


def test_normalised_diagonal_is_constant_where_the_group_average_peaks():
  """The point of the construction: the group average's prior variance peaks on the tied stratum,
  making EI's exploration term maximal on the all-identical batch. Normalising flattens it."""
  tied = np.concatenate([np.full(4, 0.4), np.full(4, 0.7)])[None, :]
  spread = np.array([[0.1, 0.4, 0.6, 0.9, 0.15, 0.35, 0.75, 0.95]])
  average, normalised = _kernel(), _normalised()
  assert average.diag(tied)[0] > 1.05 * average.diag(spread)[0]
  assert np.isclose(normalised.diag(tied)[0], 1.3) and np.isclose(normalised.diag(spread)[0], 1.3)
  assert np.all(normalised.grad_diag(spread[0]) == 0.0)


def test_normalised_is_exactly_invariant_and_psd():
  kernel, rng = _normalised(), np.random.default_rng(1)
  X, Y = rng.random((4, 8)), rng.random((3, 8))
  base = kernel(X, Y)
  for order in [(1, 0, 2, 3), (3, 2, 1, 0), (2, 3, 0, 1)]:
    assert np.allclose(kernel(_permute(X, order), Y), base, atol=1e-14)
    assert np.allclose(kernel(X, _permute(Y, order)), base, atol=1e-14)
  assert np.linalg.eigvalsh(kernel(X))[0] > -1e-12


def test_normalised_hyperparameter_gradient_matches_finite_differences():
  """The quotient rule through a normalisation is where a hand-derived gradient goes wrong, and
  sklearn's ``fit`` will not complain -- it just converges somewhere else."""
  kernel, rng = _normalised(), np.random.default_rng(2)
  X = rng.random((5, 8))
  _, gradient = kernel(X, None, True)
  for i in range(kernel.theta.size):
    step = np.zeros_like(kernel.theta)
    step[i] = 1e-6
    high, low = kernel.clone_with_theta(kernel.theta + step), kernel.clone_with_theta(kernel.theta - step)
    assert np.allclose(gradient[:, :, i], (high(X) - low(X)) / 2e-6, atol=1e-6)


def test_normalised_input_gradient_matches_finite_differences():
  kernel, rng = _normalised(), np.random.default_rng(3)
  X, x = rng.random((6, 8)), rng.random(8)
  _, jac = kernel.k_and_grad_x(X, x)
  numeric = np.zeros_like(jac)
  for j in range(8):
    step = np.zeros(8)
    step[j] = 1e-6
    numeric[:, j] = (kernel((x + step)[None, :], X)[0] - kernel((x - step)[None, :], X)[0]) / 2e-6
  assert np.allclose(jac, numeric, atol=1e-7)


def test_normalised_is_selectable_from_config():
  assert "normalised-invariant-rbf" in __kernels__


def test_normalised_refuses_a_cross_gradient_like_its_base():
  """The gradient is only defined for the symmetric call. Forwarding ``Y`` to the base keeps its
  guard; hard-coding ``None`` swallowed it and returned k(X, X) and its gradient whenever the caller
  passed a Y of the same length -- silent whenever the shapes happened to match."""
  kernel, rng = _normalised(), np.random.default_rng(4)
  X, Y = rng.random((5, 8)), rng.random((5, 8))
  with pytest.raises(ValueError, match="gradient can only be evaluated"):
    kernel(X, Y, True)


def test_normalised_repr_reports_per_group_lengthscales():
  """The repr is what sklearn prints and what lands in run logs. This kernel's point is that a
  symmetric function cannot tell experiment 1 from experiment 3, so it has one lengthscale per
  BLOCK; printing ``coordinate_length_scales()`` would show eight and assert the opposite."""
  kernel = _normalised()
  assert kernel.n_length_scales == 2
  assert repr(kernel).count(" ") < 20 and "blocks=2x4" in repr(kernel)
  assert len(np.atleast_1d(kernel._length_scales())) == kernel.n_length_scales
