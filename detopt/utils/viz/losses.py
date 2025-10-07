import numpy as np

import matplotlib.pyplot as plt

__all__ = [
  'plot'
]

def plot(losses, axes):
  n_q = 4

  if isinstance(losses, (list, tuple)):
    losses = {
      f'{i}': ls
      for i, ls in enumerate(losses)
    }
    no_label = len(losses) == 1

  elif isinstance(losses, dict):
    no_label = False
  else:
    losses = {'_': losses}
    no_label = True

  for i, key in enumerate(losses):
    ls = losses[key]
    ls = np.reshape(ls, shape=(ls.shape[0], -1))

    median = np.median(ls, axis=1)
    mean = np.mean(ls, axis=1)
    mean_error = np.std(ls, axis=1) / np.sqrt(1 + ls.shape[1])
    quantiles = np.quantile(ls, axis=1, q=np.linspace(0, 1, num=2 * n_q + 3)[1:-1])
    iters = np.arange(ls.shape[0])

    prefix = '' if no_label else f'{key} '
    for j in range(n_q):
      axes.fill_between(iters, quantiles[j], quantiles[-j - 1], color=plt.cm.tab10(i), alpha=1 / n_q)
    axes.plot(iters, median, color=plt.cm.tab10(i), label=f'{prefix}median ({median[-1]:.3f})')
    axes.plot(iters, mean, linestyle='--', color=plt.cm.tab10(i), label=f'{prefix}mean ({mean[-1]:.3f} +- {mean_error[-1]:.3f})')
    axes.legend()