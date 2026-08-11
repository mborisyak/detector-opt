"""Bayesian optimisation utilities for design search.

Provides:
  * :class:`~detopt.bo.BayesianOptimizer` — the driver everything actually uses.
    Its surrogate is a scikit-learn ARD-RBF Gaussian process fitted by marginal
    likelihood, with an analytic Expected-Improvement acquisition maximised by a
    random sweep plus exact-gradient L-BFGS-B.
  * :mod:`detopt.bo.jax_gp` — a JAX ARD-RBF GP with predictive k-fold-CV
    hyperparameter fitting and its own EI optimiser. **Reference only, for tests
    and benchmarks**: it is an independent, differentiable second opinion (used to
    check the driver's analytic EI gradient against autodiff) and the subject of
    ``scripts/benchmark_gp.py``. It is jitted, so it compiles once per distinct
    dataset size and retains every executable — do not use it in a real run.
"""

from . import jax_gp
from .jax_gp import (
    GPHParams,
    GPState,
    expected_improvement,
    fit_gp,
    gp_factorize,
    gp_posterior,
    gp_predict,
    optimise_ei,
    rbf_kernel,
    refit_gp,
)
from .bayesian_optimization import BayesianOptimizer
from .kernels import ARDRBF, NormalisedInvariantRBF, PermutationInvariantRBF, SortingRBF

__all__ = [
    "BayesianOptimizer",
    "ARDRBF",
    "PermutationInvariantRBF",
    "__kernels__",
    "SortingRBF",
    "NormalisedInvariantRBF",
    "kernel_from_config",
    # Reference JAX GP -- tests and benchmarks only (see the module docstring).
    "jax_gp",
    "GPHParams",
    "GPState",
    "fit_gp",
    "refit_gp",
    "gp_factorize",
    "gp_predict",
    "gp_posterior",
    "rbf_kernel",
    "expected_improvement",
    "optimise_ei",
]


# --------------------------------------------------------------------------- #
# GP kernel registry. Same shape as ``detopt.detector.__detectors__``: the config names one entry
# and supplies its arguments. The entry keeps the modelling CHOICE -- "does this design have a
# symmetry the surrogate should respect, and obtained how?" -- in the config, rather than implied
# by a detector's field shapes.
# --------------------------------------------------------------------------- #
__kernels__: dict[str, type] = {
    # No symmetry: sklearn's own ConstantKernel * RBF, one lengthscale per coordinate. A class of
    # its own, NOT `permutation-invariant-rbf` with no blocks -- that route agrees to machine
    # precision (there is a test) but reaches the answer by averaging over a group of one, which
    # asserts a set structure the straw/stereo designs do not have.
    "ard-rbf": ARDRBF,
    "permutation-invariant-rbf": PermutationInvariantRBF,
    # Same invariance as `permutation-invariant-rbf`, obtained by SORTING the elements rather than
    # averaging over their permutations. Constant diagonal (so EI is not drawn to degenerate
    # all-identical batches), ARD on ORDER STATISTICS rather than tied across slots, and O(m log m)
    # instead of O(2^m m). Creased on the tie locus, which is measure zero. See SortingRBF.
    "sorting-rbf": SortingRBF,
    # The same group average as `permutation-invariant-rbf`, rescaled to a CONSTANT diagonal by
    # k(x,y) / sqrt(k(x,x) k(y,y)). Keeps exact invariance AND smoothness across the tie locus
    # (which sorting gives up), while removing the |G|-fold prior-variance peak on tied designs that
    # makes EI's exploration term maximal on the all-identical batch. See NormalisedInvariantRBF.
    "normalised-invariant-rbf": NormalisedInvariantRBF,
}


def _exchangeable_blocks(detector, size):
    """Flat design indices of each exchangeable block: one block per ``design_spec`` FIELD whose
    length matches ``size``, in ``flatten_design`` order.

    For the enzyme batch with ``size = n_experiments`` that is ``enzyme_fraction`` and
    ``temperature`` -- two blocks permuted TOGETHER, because experiment ``k`` is the pair."""
    import numpy as np

    blocks, offset = [], 0
    for leaf in detector.design_spec():
        width = int(np.prod(leaf.shape))
        if width == int(size):
            blocks.append(tuple(range(offset, offset + width)))
        offset += width
    if len(blocks) == 0:
        raise ValueError(f"no design field of length {size} to permute; check `exchangeable`")
    return tuple(blocks)


def kernel_from_config(config, detector, gp):
    """Build the GP kernel named in ``config`` (``{name: {arguments}}``).

    The DIMENSION and the exchangeable blocks come from the detector, not the config: only the
    modelling choice is configurable. Hyperparameter bounds are read from the surrounding ``gp``
    block (``log_*_prior_bounds``) so swapping the kernel cannot silently change the priors.
    """
    import numpy as np

    from ..utils.config import extract

    clazz, arguments = extract(config, library=__kernels__)
    (name,) = config.keys()
    arguments = dict(arguments or {})

    # The layout kwarg is the ONLY thing that differs between the three: `ard-rbf` takes no group,
    # the group-averaged variants take `blocks`, and `sorting-rbf` takes the same index tuples as
    # `sort_blocks` -- they say which coordinates travel together under the sort, not which are
    # averaged over.
    exchangeable = arguments.pop("exchangeable", None)
    if name == "ard-rbf":
        if exchangeable is not None:
            raise ValueError("`ard-rbf` models no symmetry -- drop `exchangeable`, or name "
                             "`permutation-invariant-rbf` instead")
        layout = {}
    elif exchangeable is None:
        # Name the kernel the CALLER asked for: this branch serves every group-averaged variant, so
        # a hard-coded name sends whoever selected the normalised one to the wrong docs.
        raise ValueError(f"`{name}` needs `exchangeable: <n>`, the number of interchangeable "
                         f"design elements (e.g. n_experiments)")
    elif name == "sorting-rbf":
        layout = {"sort_blocks": _exchangeable_blocks(detector, exchangeable),
                  "key": arguments.pop("key", -1)}
    else:
        layout = {"blocks": _exchangeable_blocks(detector, exchangeable)}
    if len(arguments) > 0:
        raise ValueError(f"unknown kernel argument(s): {sorted(arguments)}")

    ls_low, ls_high = gp["log_lengthscale_prior_bounds"]
    amp_low, amp_high = gp["log_amplitude_prior_bounds"]
    return clazz(
        d=int(detector.design_dim()),
        **layout,
        constant_value=float(np.exp(amp_low + amp_high)),
        constant_value_bounds=(float(np.exp(2.0 * amp_low)), float(np.exp(2.0 * amp_high))),
        length_scale=float(np.exp(0.5 * (ls_low + ls_high))),
        length_scale_bounds=(float(np.exp(ls_low)), float(np.exp(ls_high))),
    )
