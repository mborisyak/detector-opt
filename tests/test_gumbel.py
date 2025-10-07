import os
import detopt
import numpy as np

import matplotlib.pyplot as plt

def test_gumbel(seed, plot_root):
  k = 128
  n = 1024
  m = 16
  lambda_ = 1000.0

  rng = np.random.default_rng(seed)

  xs = rng.poisson(lambda_, size=(k, n, ))
  indx = np.argsort(xs, axis=1)
  xs_sorted = np.take_along_axis(xs, indx, axis=1)
  top = xs_sorted[:, -m:]

  print(top.shape)

  approx = np.stack([
    detopt.detector.straw_signal.max_poisson(rng, lambda_, n, m)
    for _ in range(k)
  ], axis=-1)

  fig = plt.figure(figsize=(9, 6))
  axes = fig.subplots()

  axes.hist(
    [xs.ravel(), top.ravel(), approx.ravel()],
    bins=20, histtype='step', density=True,
    label=['general', 'precise', 'gumbel']
  )
  axes.legend()
  fig.savefig(os.path.join(plot_root, 'gumbel.png'))
  plt.close(fig)