import numpy as np

import matplotlib.pyplot as plt

__all__ = [
  'losses'
]

def plot_losses(losses, file):
  n_q = 4
  fig = plt.figure(figsize=(9, 6))
  axes = fig.subplots()

  if isinstance(losses, (list, tuple)):
    losses = np.stack(losses, axis=0)

  losses = np.reshape(losses, shape=(losses.shape[0], -1))

  median = np.median(losses, axis=1)
  print(np.linspace(0, 1, num=2 * n_q + 3)[1:-1])
  quantiles = np.quantile(losses, axis=1, q=np.linspace(0, 1, num=2 * n_q + 3)[1:-1])
  iters = np.arange(losses.shape[0])

  for i in range(n_q):
    axes.fill_between(iters, quantiles[i], quantiles[-i - 1], color=plt.cm.tab10(0), alpha=1 / n_q)
  axes.plot(iters, median, color=plt.cm.tab10(0), label='median')
  axes.set_yscale('log')
  axes.legend()

  fig.savefig(file)
  plt.close(fig)