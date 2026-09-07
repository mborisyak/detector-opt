"""All linear probes on d2n3 / seed 160541601 / continue / norewind: results table and one figure.
Row 1: first design's train (solid) / validation (dashed) loss per epoch, window on the right axis, one panel per probe.
Row 2: gap = |val-train| + err against window (all designs, all probes); best-so-far reported loss against cumulative
detector calls per probe. Row 3: reported loss against the closed-form Bayes risk of the design (optimism check), and
calls per design against its Bayes risk. Probe outputs were copied from the job tmp dir into output/archive/probe-linear-ladder-2026-09-05
once; the copy step is skipped for tags already present there, so the tmp path may no longer exist."""
import glob
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

root = "/home/max/dev/detector-opt"
tmp = "/home/max/.claude/jobs/e05d16e4/tmp"
keep = f"{root}/output/archive/probe-linear-ladder-2026-09-05"
TAGS = ["warmup16", "2x", "4x", "8x", "celu-2x", "n16k", "p01", "n4k", "n8k", "n4k-meta", "n8k-meta",
        "settle-continue", "settle-meta", "settle8k-continue", "settle8k-meta"]
LABEL = {
  "warmup16": "1x, warmup 16 (leaky-tanh)", "2x": "2x (leaky-tanh)", "4x": "4x (leaky-tanh)", "8x": "8x (leaky-tanh)",
  "celu-2x": "2x, celu", "n16k": "n0 16384 / inc 8192, celu", "p01": "precision 0.1, 2048/1024, celu",
  "n4k": "4096/2048, rewind 0.5, celu", "n8k": "8192/4096, rewind 0.5, celu",
  "n4k-meta": "META 4096/2048, rewind 0.5, celu", "n8k-meta": "META 8192/4096, rewind 0.5, celu",
  "settle-continue": "SETTLE-FIRST 4096/2048 rw0.5", "settle-meta": "SETTLE-FIRST META 4096/2048 rw0.5",
  "settle8k-continue": "SETTLE-FIRST 8192/4096 rw0.5", "settle8k-meta": "SETTLE-FIRST META 8192/4096 rw0.5",
}
noise, d, m = 0.5, 2, 3

def bayes_risk(x_scaled):
  flat = -1.0 + 2.0 * np.asarray(x_scaled, np.float64)
  probe = flat.reshape(d, m).T
  rows = np.concatenate([probe, np.ones((m, 1))], axis=-1)
  return float(np.trace(np.linalg.inv(rows.T @ rows / noise**2 + np.eye(d + 1))) / (d + 1))

os.makedirs(keep, exist_ok=True)
for tag in TAGS:
  src = f"{tmp}/probe-{tag}/d2n3-160541601-continue-norewind"
  if os.path.isdir(src) and not os.path.isdir(f"{keep}/{tag}"):
    shutil.copytree(src, f"{keep}/{tag}", ignore=shutil.ignore_patterns("checkpoints"))

probes = {}
for tag in TAGS:
  p = f"{keep}/{tag}"
  if not os.path.isdir(p):
    continue
  cfg = yaml.safe_load(open(f"{p}/config.yaml"))
  t = cfg["training"]
  act = cfg["regressor"]["set-regressor"].get("activation", "leaky-tanh")
  hs = sorted(glob.glob(f"{p}/plots/iter_*_history.npz"))
  rows = json.load(open(f"{p}/results.json"))["results"] if os.path.exists(f"{p}/results.json") else []
  rows = [x for x in rows if x.get("loss") is not None]
  designs = []
  for k, h in enumerate(hs):
    z = np.load(h)
    tr, va, w = z["train_loss_per_epoch"], z["val_loss_per_epoch"], z["train_budget_per_epoch"]
    gap = np.abs(va - tr) + np.hypot(z["train_sem_per_epoch"], z["val_sem_per_epoch"])
    row = rows[k] if k < len(rows) else None
    designs.append(dict(
      tr=tr, va=va, w=w, gap=gap, epochs=len(tr), additions=int(np.sum(np.diff(w) != 0)), window=int(w[-1]),
      spent=row["spent"] if row else None, loss=row["loss"] if row else None,
      bayes=bayes_risk(row["x_scaled"]) if row else None, complete=row is not None,
    ))
  probes[tag] = dict(cfg=t, act=act, designs=designs)
  print(f"\n### {tag}: {LABEL[tag]}  | n0 {t['n0']} inc {t['n_increment']} IL {t['iteration_limit']} budget {t['budget']} "
        f"precision {t['loss_precision']} warmup {t['warmup_epochs']} activation {act}")
  print(f"  designs completed: {len(rows)}" + ("" if len(hs) == len(rows) else f" (+{len(hs) - len(rows)} unfinished)"))
  print(f"  {'#':>3}{'window':>8}{'adds':>6}{'epochs':>7}{'spent':>8}{'loss':>9}{'bayes':>8}{'loss-bayes':>11}{'gap@exit':>10}")
  for k, g in enumerate(designs):
    if not g["complete"]:
      print(f"  {k+1:>3}{g['window']:>8}{g['additions']:>6}{g['epochs']:>7}{'-':>8}{'-':>9}{'-':>8}{'-':>11}{g['gap'][-1]:>10.4f}  (cut / capped)")
      continue
    print(f"  {k+1:>3}{g['window']:>8}{g['additions']:>6}{g['epochs']:>7}{g['spent']:>8}{g['loss']:>9.4f}{g['bayes']:>8.4f}"
          f"{g['loss'] - g['bayes']:>+11.4f}{g['gap'][-1]:>10.4f}")
  done = [g for g in designs if g["complete"]]
  if len(done) > 0:
    print(f"  mean spent/design {np.mean([g['spent'] for g in done]):.0f}, mean epochs/design {np.mean([g['epochs'] for g in done]):.0f}, "
          f"best loss {min(g['loss'] for g in done):.4f}, mean loss-bayes {np.mean([g['loss'] - g['bayes'] for g in done]):+.4f}")

colors = dict(zip(TAGS, ["0.4", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "gold", "navy", "darkred", "teal", "magenta"]))
fig = plt.figure(figsize=(36, 16))
gs = fig.add_gridspec(3, len(probes), height_ratios=[1.0, 1.1, 1.0])
for j, (tag, P) in enumerate(probes.items()):
  ax = fig.add_subplot(gs[0, j])
  g = P["designs"][0]
  ep = np.arange(1, g["epochs"] + 1)
  ax.plot(ep, g["tr"], color=colors[tag], lw=1.0, label="train")
  ax.plot(ep, g["va"], color=colors[tag], lw=1.0, ls="--", label="validation")
  ax.set_ylim(0.05, 0.6)
  ax.set_title(f"{LABEL[tag]}\nfirst design: {g['epochs']} epochs, {g['additions']} additions\nwindow -> {g['window']}", fontsize=9)
  ax.set_xlabel("epoch", fontsize=8)
  ax.grid(alpha=0.3)
  ax.tick_params(labelsize=8)
  ax2 = ax.twinx()
  ax2.plot(ep, g["w"], color="0.65", lw=0.8)
  ax2.set_yscale("log")
  ax2.tick_params(axis="y", colors="0.5", labelsize=7)
  if j == 0:
    ax.set_ylabel("loss")
    ax.legend(fontsize=8, loc="upper right")
n = len(probes)
ax_gap = fig.add_subplot(gs[1, : max(1, n // 2)])
ax_bo = fig.add_subplot(gs[1, max(1, n // 2):])
ax_opt = fig.add_subplot(gs[2, : max(1, n // 2)])
ax_cost = fig.add_subplot(gs[2, max(1, n // 2):])
for tag, P in probes.items():
  for k, g in enumerate(P["designs"]):
    ends = np.append(np.flatnonzero(np.diff(g["w"]) != 0), len(g["w"]) - 1)
    ax_gap.plot(g["w"][ends], g["gap"][ends], color=colors[tag], lw=1.2 if k == 0 else 0.6, alpha=1.0 if k == 0 else 0.45,
                marker="." if k == 0 else None, ms=3, label=LABEL[tag] if k == 0 else None)
  done = [g for g in P["designs"] if g["complete"]]
  if len(done) > 0:
    calls = np.cumsum([g["spent"] for g in done])
    best = np.minimum.accumulate([g["loss"] for g in done])
    ax_bo.step(calls, best, where="post", color=colors[tag], lw=1.4, marker="o", ms=3, label=f"{LABEL[tag]} ({len(done)} designs)")
    ax_opt.scatter([g["bayes"] for g in done], [g["loss"] for g in done], color=colors[tag], s=22, label=LABEL[tag], zorder=3)
    ax_cost.scatter([g["bayes"] for g in done], [g["spent"] for g in done], color=colors[tag], s=22, label=LABEL[tag], zorder=3)
ax_gap.axhline(0.02, color="red", ls=":", lw=1, label="precision 0.02")
ax_gap.axhline(0.1, color="red", ls="--", lw=0.8, label="precision 0.1")
ax_gap.set_xscale("log"); ax_gap.set_yscale("log")
ax_gap.set_xlabel("training window at round end (rows)"); ax_gap.set_ylabel("gap = |val - train| + err")
ax_gap.set_title("gap against window, all designs (thin = later designs)", fontsize=10)
ax_gap.grid(alpha=0.3, which="both"); ax_gap.legend(fontsize=7, ncol=2)
ax_bo.axhline(0.0937, color="k", ls=":", lw=1, label="closed-form optimum 0.0937")
ax_bo.set_xscale("log")
ax_bo.set_xlabel("cumulative detector calls"); ax_bo.set_ylabel("best reported loss so far")
ax_bo.set_title("BO trajectory per probe (budget 262144 unless 4x 524288 / 8x 1048576)", fontsize=10)
ax_bo.grid(alpha=0.3, which="both"); ax_bo.legend(fontsize=7)
lo, hi = 0.05, 0.6
ax_opt.plot([lo, hi], [lo, hi], color="k", lw=0.8, ls=":", label="reported = Bayes risk")
ax_opt.set_xlabel("closed-form Bayes risk of the design"); ax_opt.set_ylabel("reported loss")
ax_opt.set_title("reported against the floor: below the line = optimism accepted by the bar", fontsize=10)
ax_opt.grid(alpha=0.3); ax_opt.legend(fontsize=7)
ax_cost.set_yscale("log")
ax_cost.set_xlabel("closed-form Bayes risk of the design"); ax_cost.set_ylabel("detector calls spent on the design")
ax_cost.set_title("spend per design", fontsize=10)
ax_cost.grid(alpha=0.3, which="both"); ax_cost.legend(fontsize=7)
fig.suptitle("linear d2n3, seed 160541601 -- every probe (continue unless META; SETTLE-FIRST = check (1) removed, IL 262144, budget 524288); adamw + cosine 4x->1x/16, same convergence procedure", fontsize=12)
fig.tight_layout()
out = f"{root}/output/plots/linear-ladder-probes-2026-09-05/all_probes.png"
fig.savefig(out, dpi=100)
print("\n" + out)
