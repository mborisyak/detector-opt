import pytest

import jax

# CPU is the default backend for tests: jax.devices() returns only CPU, so any
# test (and library code defaulting to jax.devices()[0]) stays on CPU. CUDA
# devices remain individually reachable via jax.devices("cuda"); the `cuda`
# fixture below opts a specific test onto the GPU.
jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(scope="function")
def cuda():
    """Run the requesting test on CUDA, skipping if no GPU is present.

    Tests that do *not* request this fixture run on CPU (the default above). A
    test that takes ``cuda`` runs its body under ``jax.default_device(gpu)`` and
    receives the device (e.g. to pass as ``device=`` to a DesignTrainer/Pool).
    """
    try:
        devices = jax.devices("cuda")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("no CUDA device available")
    with jax.default_device(devices[0]):
        yield devices[0]


@pytest.fixture(scope="function")
def plot_root(request):
    import os, pathlib

    f = request.function
    here, _ = os.path.split(__file__)
    root = os.path.join(here, "plots", f.__name__)
    os.makedirs(root, exist_ok=True)

    return pathlib.Path(root)


@pytest.fixture(scope="function")
def seed(request):
    import hashlib

    h = hashlib.sha256()
    h.update(bytes(request.function.__name__, encoding="utf-8"))
    digest = h.hexdigest()

    return int(digest[:8], 16)
