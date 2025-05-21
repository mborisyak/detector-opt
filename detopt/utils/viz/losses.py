import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

__all__ = [
  'losses'
]

DEFAULT_FIGURE_SIZE = (9, 6)

def plot_losses(losses, axes: plt.Axes, color=0, n_q: int | None=4, label=None):
  if isinstance(losses, (list, tuple)):
    losses = np.stack(losses, axis=0)

  losses = np.reshape(losses, shape=(losses.shape[0], -1))

  if isinstance(color, int):
    color = plt.cm.tab10(color)

  iters = np.arange(losses.shape[0])
  median = np.median(losses, axis=1)
  axes.plot(iters, median, color=color, label=label)

  if n_q is not None:
    qs = np.linspace(0, 1, num=2 * n_q + 3)[1:-1]
    quantiles = np.quantile(losses, axis=1, q=qs)
    for i in range(n_q):
      axes.fill_between(iters, quantiles[i], quantiles[-i - 1], color=color, alpha=1 / n_q)

  if label is not None:
    axes.legend()

  return axes