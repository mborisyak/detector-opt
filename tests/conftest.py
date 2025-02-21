import pytest

import jax; jax.config.update('jax_platform_name', 'cpu')

@pytest.fixture(scope='function')
def plot_root(request):
  import os
  f = request.function
  here, _ = os.path.split(__file__)
  root = os.path.join(here, 'plots', f.__name__)
  os.makedirs(root, exist_ok=True)

  return root

@pytest.fixture(scope='function')
def seed(request):
  import hashlib

  h = hashlib.sha256()
  h.update(bytes(request.function.__name__, encoding='utf-8'))
  digest = h.hexdigest()

  return int(digest[:8], 16)
