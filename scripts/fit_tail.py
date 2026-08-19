#!/usr/bin/env python3
"""Fit `A * exp(alpha * t) + c` to the training curve after the LAST data addition.

    python scripts/fit_tail.py output/criterion/plots/bayes/seed1/iter_008_history.npz

`c` is where the curve was heading. `final - c` is how much descent was left when the run stopped.
`alpha` is the rate, and `-1/alpha` the epochs the remaining descent would have taken to fall by 1/e.
"""

import argparse

import numpy as np
import scipy.optimize


def last_round(history):
  """Indices of the epochs after the final data addition."""
  budget = history["train_budget_per_epoch"]
  changed = np.flatnonzero(np.diff(budget) != 0)
  start = int(changed[-1]) + 1 if changed.size > 0 else 0
  return start, np.arange(start, budget.size)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("history", nargs="+")
  arguments = parser.parse_args()

  for path in arguments.history:
    history = np.load(path)
    start, index = last_round(history)
    loss = np.asarray(history["train_loss_per_epoch"], np.float64)[index]
    t = np.arange(loss.size, dtype=np.float64)
    if loss.size < 4:
      print(f"{path}: only {loss.size} epochs after the last addition -- not fitted")
      continue
    guess = (loss[0] - loss[-1], -1.0 / max(loss.size / 3.0, 1.0), loss[-1])
    try:
      (a, alpha,
       c), _ = scipy.optimize.curve_fit(lambda t, a, alpha, c: a * np.exp(alpha * t) + c, t, loss, p0=guess, maxfev=20000)
    except RuntimeError as error:
      print(f"{path}: fit failed ({error})")
      continue
    residual = float(np.sqrt(np.mean((a * np.exp(alpha * t) + c - loss)**2)))
    window = int(history["train_budget_per_epoch"][-1])
    tail = f"   1/e time {-1.0 / alpha:.1f} epochs" if alpha < 0 else "   NOT DECAYING"
    print(f"{path}")
    print(f"  last addition at epoch {start}, {loss.size} epochs after it, window {window}")
    print(f"  A {a:+.5f}  alpha {alpha:+.5f}  c {c:.5f}   rms residual {residual:.6f}")
    print(f"  final {loss[-1]:.5f}   asymptote {c:.5f}   REMAINING {loss[-1] - c:+.5f}{tail}")


if __name__ == "__main__":
  main()
