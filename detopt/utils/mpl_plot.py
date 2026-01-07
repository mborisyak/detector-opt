import numpy as np

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

__all__ = [
  'get_rows_cols',
  'losses'
]

def get_rows_cols(n: int):
  import math

  best_error = n * n
  best_cols = None
  best_rows = None

  for n_cols in range(1, int(math.sqrt(n)) + 1):
    n_rows = n // n_cols + (0 if n % n_cols == 0 else 0)
    error = abs(n_rows * n_cols - n) + 0.1 * abs(n_rows * n_rows - n)
    if error < best_error:
      best_error = error
      best_cols = n_cols
      best_rows = n_rows

  return best_rows, best_cols

def plot_loss(loss, axes):
  n_q = 4

  if isinstance(loss, (list, tuple)):
    loss = {
      f'{i}': ls
      for i, ls in enumerate(loss)
    }
    no_label = len(loss) == 1

  elif isinstance(loss, dict):
    no_label = False
  else:
    loss = {'_': loss}
    no_label = True

  for i, key in enumerate(loss):
    ls = loss[key]
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

def _plot_losses(fname: str | bytes, *loss_infos):
  fig = plt.figure(figsize=(9, 6 * len(loss_infos)))
  axes = fig.subplots(len(loss_infos), 1, squeeze=False).ravel()

  for i, loss_info in enumerate(loss_infos):
    if 'title' in loss_info:
      axes[i].set_title(loss_info['title'])
    if 'label' in loss_info:
      axes[i].set_ylabel(loss_info['label'])

    plot_loss(loss_info['losses'], axes[i])

  fig.tight_layout()
  fig.savefig(fname)
  plt.close(fig)

def losses(fname: str | bytes, *loss_infos):
  import threading
  return threading.Thread(target=_plot_losses, args=(fname, *loss_infos)).start()

