"""Architecture/ablation comparison at one seed: best loss per arm, one series per variant.

One panel per measured quantity. Cells still in flight are drawn hollow and annotated with the
fraction of budget spent, because a partial cell's best-so-far can only fall further -- reading a
hollow marker as a final value is the mistake this annotation exists to prevent.
"""
import argparse, json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["from_scratch", "continue", "closest", "meta"]
COLOURS = {"baseline": "#4E79A7", "w2x": "#E15759", "w15x_deep": "#59A14F", "b3taper": "#B07AA1",
           "norewind": "#EDC948", "w2x_2M": "#9C2B2D", "w2x_2M_norw": "#D4A017", "b3taper_2M": "#6A3D7A"}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", default="/tmp/capdata.json")
  ap.add_argument("--precision", type=float, default=0.01)
  ap.add_argument("--out", default="/tmp/render/capacity_probe.png")
  args = ap.parse_args()
  rows = json.load(open(args.data))
  variants = [
    v for v in ["baseline", "norewind", "w2x", "w15x_deep", "b3taper", "w2x_2M", "w2x_2M_norw", "b3taper_2M"]
    if any(r["variant"] == v and r["best"] is not None for r in rows)
  ]

  fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
  base = {r["arm"]: r["best"] for r in rows if r["variant"] == "baseline"}

  for ax, key, title, ylab in [(axes[0], "best", "Best loss reached (lower is better)", "best loss"),
                               (axes[1], "spent", "Detector calls per design (median)", "calls per design")]:
    for vi, v in enumerate(variants):
      xs, ys, hollow = [], [], []
      for ai, a in enumerate(ARMS):
        r = next((q for q in rows if q["variant"] == v and q["arm"] == a), None)
        if r is None or r[key] is None:
          continue
        xs.append(ai + (vi - (len(variants) - 1) / 2) * 0.15)
        ys.append(r[key])
        hollow.append(r["status"] != "COMPLETED")
      if not xs:
        continue
      c = COLOURS.get(v, "#888888")
      for x, y, h in zip(xs, ys, hollow):
        ax.plot([x], [y], "o", ms=11, mfc=("none" if h else c), mec=c, mew=2.2, zorder=3)
      ax.plot(xs, ys, "-", color=c, lw=1.6, alpha=0.55, zorder=2, label=v)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.25, lw=0.7)
    ax.set_axisbelow(True)

  if base:
    lo = min(v for v in base.values())
    axes[0].axhspan(lo - args.precision, lo + args.precision, color="#4E79A7", alpha=0.08, zorder=0)
    axes[0].annotate(
      f"+-loss_precision ({args.precision}) around the baseline's best arm", xy=(0.02, lo + args.precision), fontsize=9,
      color="#4E79A7", va="bottom"
    )

  for r in rows:
    if r["best"] is not None and r["status"] != "COMPLETED" and r["frac"] is not None:
      ai = ARMS.index(r["arm"])
      vi = variants.index(r["variant"])
      axes[0].annotate(
        f"{100*r['frac']:.0f}% budget", xy=(ai + (vi - (len(variants) - 1) / 2) * 0.15, r["best"]), xytext=(0, -16),
        textcoords="offset points", ha="center", fontsize=8, color=COLOURS.get(r["variant"])
      )

  for ax in axes:
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.08 * (hi - lo), hi)
  axes[0].legend(frameon=False, fontsize=10, loc="upper right")
  ncap = sum(1 for r in rows if r.get("capped"))
  fig.suptitle(
    "SHiP capacity / rewind probes -- seed 1244111331, ladder 16384/8192\n"
    "solid = COMPLETED, hollow = still running (best-so-far can only fall further)" +
    (f"   |   {ncap} CAPPED cell(s) at this seed" if ncap > 0 else ""), fontsize=12
  )
  fig.tight_layout(rect=(0, 0, 1, 0.93))
  fig.savefig(args.out, dpi=130)
  print(f"  wrote {args.out}")


if __name__ == "__main__":
  main()
