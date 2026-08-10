import numpy as np
import jax
import jax.numpy as jnp
import pytest

from flax import nnx

import detopt

# The legacy `detopt.optimizer.Subgradient` (+ AlphaResNet) predates the typed-records detector API
# (Event/Target/Design namedtuples, combine/combine_scaled). It is superseded by the standalone
# design-optimization loop in scripts/subgradient.py; this test is skipped rather than ported.
pytestmark = pytest.mark.skip(reason="deprecated detopt.optimizer.Subgradient path; use scripts/subgradient.py")


# Test design (the detector holds none): flat [station_z(4), view_tilt(4), field_strength].
_DEBUG_DESIGN = np.array([225.0, 250.0, 350.0, 375.0, -12.0, -4.0, 4.0, 12.0, 0.3], np.float32)


def test_subgradient(seed):
    rngs = nnx.Rngs(seed + 1)

    detector = detopt.detector.DebugDetector()
    regressor = detopt.nn.AlphaResNet(detector, n_hidden=32, depth=3, rngs=rngs)

    opt = detopt.optimizer.Subgradient(detector, regressor, batch_size=7, n_steps_regressor=3)

    design = _DEBUG_DESIGN

    for i in range(100):
        design = opt.step(seed=seed + i, design=design)

    print(design)
