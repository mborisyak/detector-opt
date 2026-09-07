"""Same rung, two trainers: gap = |val-train| + err at round ends versus window size, d2n3 `continue` cells.
Campaign (adamw + cosine 4x, warmup 2, norewind) against the Sep-3 archive (adamaxw 1x, no cosine, warmup 2).
Also the first 8 epochs of the first round, and where the two evaluation functions live."""
import glob
import subprocess

import numpy as np

root = "/home/max/dev/detector-opt"
sets = {
  "campaign adamw+cos": sorted(glob.glob(f"{root}/output/linear/select/d2n3/*/continue/norewind/plots/iter_000_history.npz")),
  "sep-3 adamaxw": sorted(glob.glob(f"{root}/output/archive/20260903T134516-linear-selection-design-capped/linear/d2n3/test/*/continue/plots/iter_000_history.npz")),
}
marks = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
for label, paths in sets.items():
  print(f"\n### {label}: {len(paths)} cells")
  for p in paths[:2]:
    h = np.load(p)
    tr, va, w = h["train_loss_per_epoch"], h["val_loss_per_epoch"], h["train_budget_per_epoch"]
    err = np.hypot(h["train_sem_per_epoch"], h["val_sem_per_epoch"])
    gap = np.abs(va - tr) + err
    adds = np.flatnonzero(np.diff(w) != 0) + 1
    ends = adds - 1
    row = []
    for m in marks:
      k = [i for i in ends if w[i] >= m]
      row.append(f"{gap[k[0]]:.3f}" if len(k) > 0 else "-")
    print(f"  seed {p.split('/')[-5]}: epochs {len(tr)}, additions {len(adds)}, final window {int(w[-1])}")
    print(f"    gap at first round-end with window >= {marks}: {row}")
    print(f"    first round train {np.round(tr[:8], 3).tolist()}")
    print(f"    first round val   {np.round(va[:8], 3).tolist()}  (round ends at epoch {adds[0] if len(adds) else '-'})")
    print(f"    err (sem) at those ends: {[f'{err[k[0]]:.4f}' for m in marks for k in [[i for i in ends if w[i] >= m]] if len(k) > 0][:6]}")
print("\n### evaluation functions")
print(subprocess.run(["grep", "-n", "-E", "def _eval_train|def _eval_val|def _train_epoch|steps_per_epoch", f"{root}/detopt/nn/trainer/design.py", f"{root}/detopt/nn/trainer/common.py"], capture_output=True, text=True).stdout[:1500])
