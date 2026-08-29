"""BO trajectories: every design evaluation and the best-so-far envelope it produces.

Only CANONICAL arm directories are read. The SHiP 1M tree contains `750143450/from_scratch.stopped-
10designs`, a stopped pre-refactor run; walking directories blindly counts it as a real cell.

One panel per campaign tree. Within a panel, one colour per arm; every (seed, arm) cell contributes
a step-wise best-so-far line and a marker at each design's OWN observed loss, so the search is
visible rather than only its running minimum. A best-so-far line alone hides how BO got there --
whether it fell early and sat flat, or kept finding improvements to the end.

Cells that CAPPED are drawn dashed and named in the panel title: a capped cell died mid-run and its
whole seed is excluded from paired analysis, so its trajectory must not be read as a result.
"""
import argparse, json, os, glob
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["from_scratch", "continue", "closest", "meta"]
ARM_COLOUR = {"from_scratch": "#4E79A7", "continue": "#E15759", "closest": "#59A14F", "meta": "#B07AA1"}
CAPPED_EMPTY = {}


def capped_without_designs(root):
  """Cells that CAPPED without banking a usable design, so they contribute no trajectory.

  Testing only for a MISSING results.json is not enough: a capped cell can write the file with no
  usable rows in it, and it then reads as absent rather than as the failure it is."""
  out = []
  for st in sorted(glob.glob(os.path.join(root, "*", "*", "status.txt"))):
    d = os.path.dirname(st)
    rj = os.path.join(d, "results.json")
    if os.path.exists(rj):
      try:
        usable = [r for r in json.load(open(rj)).get("results", []) if r.get("loss") is not None and r.get("spent") is not None]
      except Exception:
        usable = []
      if len(usable) > 0:
        continue
    if d.split(os.sep)[-1] not in ARMS:
      continue
    if "CAPPED" in open(st, errors="ignore").read():
      out.append(f"{d.split(os.sep)[-2]}/{d.split(os.sep)[-1]}")
  return out


def load_tree(root):
  cells = []
  for f in sorted(glob.glob(os.path.join(root, "*", "*", "results.json"))):
    seed, arm = f.split(os.sep)[-3], f.split(os.sep)[-2]
    if arm not in ARMS:
      continue
    try:
      j = json.load(open(f))
    except Exception:
      continue
    rows = [r for r in j.get("results", []) if r.get("loss") is not None and r.get("spent") is not None]
    if len(rows) == 0:
      continue
    rows.sort(key=lambda r: r.get("iteration", 0))
    spent = np.cumsum([r["spent"] for r in rows])
    loss = np.asarray([float(r["loss"]) for r in rows])
    d = os.path.dirname(f)
    capped = os.path.exists(os.path.join(d, "CAPPED.txt"))
    status = "-"
    st = os.path.join(d, "status.txt")
    if os.path.exists(st):
      for line in open(st, errors="ignore"):
        if line.startswith("status="):
          status = line.strip().split("=", 1)[1]
    cells.append(
      dict(
        seed=seed, arm=arm, spent=spent, loss=loss, capped=capped, status=status,
        budget=j.get("config", {}).get("training", {}).get("budget")
      )
    )
  return cells


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("trees", nargs="+")
  ap.add_argument("--out", required=True)
  ap.add_argument("--title", default="BO trajectories")
  args = ap.parse_args()

  loaded = [(os.path.basename(t.rstrip("/")), load_tree(t)) for t in args.trees]
  global CAPPED_EMPTY
  CAPPED_EMPTY = {os.path.basename(t.rstrip("/")): capped_without_designs(t) for t in args.trees}
  loaded = [(n, c) for n, c in loaded if len(c) > 0]
  if len(loaded) == 0:
    raise SystemExit("no tree has any banked design")

  ncol = min(3, len(loaded))
  nrow = int(np.ceil(len(loaded) / ncol))
  fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.9 * nrow), squeeze=False)

  ylo = min(c["loss"].min() for _, cells in loaded for c in cells)
  yhi = np.percentile(np.concatenate([c["loss"] for _, cells in loaded for c in cells]), 97)

  for k, (name, cells) in enumerate(loaded):
    ax = axes[k // ncol][k % ncol]
    seeds = sorted(set(c["seed"] for c in cells))
    budget = next((c["budget"] for c in cells if c["budget"] is not None), None)
    for c in cells:
      col = ARM_COLOUR[c["arm"]]
      bsf = np.minimum.accumulate(c["loss"])
      ax.step(
        c["spent"], bsf, where="post", color=col, lw=2.0 if len(seeds) == 1 else 1.1, alpha=1.0 if len(seeds) == 1 else 0.55,
        ls="--" if c["capped"] else "-", zorder=3
      )
      ax.plot(c["spent"], c["loss"], "o", ms=4.5, mfc="none", mec=col, mew=1.0, alpha=0.75, zorder=2)
    if budget is not None:
      ax.axvline(budget, color="#666666", lw=1.0, ls=":", zorder=1)
      ax.annotate(
        "budget", xy=(budget, yhi), xytext=(-4, -10), textcoords="offset points", ha="right", fontsize=8, color="#666666"
      )
    capped = [f"{c['seed']}/{c['arm']}" for c in cells if c["capped"]] + CAPPED_EMPTY.get(name, [])
    ax.set_title(
      f"{name}\n{len(cells)} cells, {len(seeds)} seed(s)" + (f"   CAPPED: {', '.join(capped)}" if capped else ""), fontsize=10
    )
    ax.set_xlabel("cumulative detector calls")
    ax.set_ylabel("loss")
    ax.set_ylim(ylo - 0.01, yhi)
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)

  for k in range(len(loaded), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")

  handles = [plt.Line2D([], [], color=ARM_COLOUR[a], lw=2.4, label=a) for a in ARMS]
  handles.append(plt.Line2D([], [], color="#666666", lw=1.6, ls="--", label="capped cell"))
  handles.append(plt.Line2D([], [], color="#666666", lw=0, marker="o", mfc="none", label="one design evaluation"))
  axes[0][0].legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
  fig.suptitle(args.title + "   --   step line = best-so-far, circles = each design's own loss", fontsize=12)
  fig.tight_layout(rect=(0, 0, 1, 0.95))
  fig.savefig(args.out, dpi=130)
  print(f"  wrote {args.out}  ({len(loaded)} panels)")


if __name__ == "__main__":
  main()
