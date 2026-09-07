"""Is the train/val gap plain optimism? For the two designs of the 4x probe: Bayes risk of the design (the floor any
regressor can reach), the converged train and val losses, and the effective parameter count implied by
gap = 2 * sigma^2 * p_eff / N at every round end (sigma^2 = the val loss, N = the training window)."""
import glob
import json

import numpy as np

root = "/home/max/dev/detector-opt/output/archive/probe-linear-ladder-2026-09-05"
noise, d, m = 0.5, 2, 3

def bayes_risk(x_scaled):
  flat = -1.0 + 2.0 * np.asarray(x_scaled, np.float64)
  probe = flat.reshape(d, m).T
  rows = np.concatenate([probe, np.ones((m, 1))], axis=-1)
  precision = rows.T @ rows / noise**2 + np.eye(d + 1)
  return float(np.trace(np.linalg.inv(precision)) / (d + 1))

for tag in ("2x", "4x", "8x"):
  r = json.load(open(f"{root}/{tag}/results.json"))
  hs = sorted(glob.glob(f"{root}/{tag}/plots/iter_*_history.npz"))
  print(f"\n### {tag}")
  for k, x in enumerate(r["results"]):
    h = np.load(hs[k])
    tr, va, w = h["train_loss_per_epoch"], h["val_loss_per_epoch"], h["train_budget_per_epoch"]
    print(f"design {k+1}: bayes risk {bayes_risk(x['x_scaled']):.4f} | converged train {tr[-1]:.4f} val {va[-1]:.4f} "
          f"reported loss {x['loss']:.4f} | train below floor by {bayes_risk(x['x_scaled']) - tr[-1]:+.4f}, val above by {va[-1] - bayes_risk(x['x_scaled']):+.4f}")
    ends = np.append(np.flatnonzero(np.diff(w) != 0), len(w) - 1)
    rows = []
    for i in ends:
      if w[i] in (8192, 16384, 32768, 65536, 131072) or i == ends[-1]:
        gap = va[i] - tr[i]
        rows.append(f"N={int(w[i])}: val-train={gap:.4f} -> p_eff={gap * w[i] / (2 * va[i]):.0f}")
    print("   " + " | ".join(rows))
