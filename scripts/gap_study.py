"""Autonomous FairShip-gap bake-off: train each (detector, regressor) candidate on an equal budget at a
FIXED design and record held-out per-component normalized MSE (R2 ~ 1 - MSE). Resumable; writes
data/gap_study/{results.json, leaderboard.md} incrementally + a checkpoint per candidate.

Run:  python scripts/gap_study.py                 # all pending candidates
      python scripts/gap_study.py only=image/conv # a single candidate
See tasks/fairship-gap-study.md for the plan."""
import os
import sys
import json
import time
import traceback

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import regression as REG  # noqa: E402

OUT = "data/gap_study"
RESULTS = os.path.join(OUT, "results.json")
LEADER = os.path.join(OUT, "leaderboard.md")

DESIGN = "config/design/initial_stereo.yaml"
SAMPLES = 32768
VAL_SAMPLES = 8192
E1 = 60  # round-1 budget

# (name, detector_config_path, regressor_config, batch, epochs)
CANDIDATES = [
    # --- R1 bake-off ---
    ("layerwise/set", "config/detector/stereo_layerwise.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, E1),
    ("layerwise/predictive", "config/detector/stereo_layerwise.yaml",
     {"predictive-set-regressor": {"features": [[128, 64], [128, 64]], "aux_weight": 0.3}}, 64, E1),
    ("layerwise/predictive-prob", "config/detector/stereo_layerwise.yaml",
     {"predictive-prob-regressor": {"features": [[128, 64], [128, 64]], "head_features": [128], "prob_weight": 0.1}}, 64, E1),
    ("layerwise/predictive-mixture", "config/detector/stereo_layerwise.yaml",
     {"predictive-mixture-regressor": {"features": [[128, 64], [128, 64]], "head_features": [128],
                                       "n_components": 4, "mixture_weight": 0.1}}, 64, E1),
    ("layerwise/masked", "config/detector/stereo_layerwise.yaml",
     {"masked-set-regressor": {"features": [[128, 64], [128, 64]], "head_features": [128], "mask_weight": 0.3}}, 64, E1),
    ("hits/continuous", "config/detector/stereo_hits.yaml",
     {"continuous-conv-regressor": {"features": [[64], [64]], "kernel_features": [64], "kernel_per_block": False}}, 8, E1),
    ("hits/continuous-pb", "config/detector/stereo_hits.yaml",
     {"continuous-conv-regressor": {"features": [[64], [48]], "kernel_features": [64], "kernel_per_block": True}}, 8, E1),
    ("image/conv", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 5, "channels": [16, 16, 32, 32, 64, 64]}}, 32, E1),
    ("stereo2/set-ref", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, E1),
    # --- R2: scale the winner (image/conv) ---
    ("image/conv-wide", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 7, "channels": [32, 32, 64, 64, 128, 128]}}, 32, 200),
    ("image/conv-long", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 5, "channels": [16, 16, 32, 32, 64, 64]}}, 32, 200),
    # --- R3: data scaling (6th element = pool size). Hypothesis: val R2 rises with more unique data. ---
    ("stereo2/set-4x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, 60, 131072),
    ("stereo2/set-8x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, 60, 262144),
    ("image/conv-4x", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 5, "channels": [16, 16, 32, 32, 64, 64]}}, 32, 60, 131072),
    ("image/conv-wide-4x", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 7, "channels": [32, 32, 64, 64, 128, 128]}}, 32, 60, 131072),
    # --- R4: push data + capacity on the hit-level winner toward the ceiling ---
    ("stereo2/set-16x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, 40, 524288),
    ("stereo2/set-32x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, 30, 1048576),
    ("stereo2/set-big-16x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 128], [256, 128], [128, 64]]}}, 64, 40, 524288),
    ("image/conv-8x", "config/detector/stereo_image.yaml",
     {"conv-regressor": {"n_stations": 4, "n_views_per_station": 4, "n_layers_per_view": 2,
                         "straw_kernel": 5, "channels": [16, 16, 32, 32, 64, 64]}}, 32, 40, 262144),
    # --- R5: push capacity (the strongest lever) at the GPU-buffer data limit ---
    ("stereo2/set-huge-16x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [128, 64]]}}, 64, 40, 524288),
    ("stereo2/set-vhuge-16x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[512, 256], [512, 256], [256, 128], [128, 64]]}}, 64, 40, 524288),
    ("stereo2/set-big-24x", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 128], [256, 128], [128, 64]]}}, 64, 40, 786432),
    # --- R7: stable DEEPER net (4/5-layer collapsed at lr=1e-3) at lower peak LR. 262k pool (524k OOM
    #         was a transient external spike); apples-to-apples vs the 3L control at the same 262k. ---
    ("stereo2/set-3L-262k", "config/detector/stereo2.yaml",   # depth control at this data/LR
     {"set-regressor": {"features": [[256, 128], [256, 128], [128, 64]]}}, 64, 50, 262144, 5.0e-4),
    ("stereo2/set-4L-lr3e4", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [128, 64]]}}, 64, 50, 262144, 3.0e-4),
    ("stereo2/set-4L-lr5e4", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [128, 64]]}}, 64, 50, 262144, 5.0e-4),
    ("stereo2/set-5L-lr3e4", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [256, 128], [128, 64]]}}, 64, 50, 262144, 3.0e-4),
    # --- R8: combined best (depth + data + tuned LR) to pin the scaling ceiling ---
    ("stereo2/set-5L-524k", "config/detector/stereo2.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [256, 128], [128, 64]]}}, 64, 50, 524288, 4.0e-4),
    # --- R9: curvature-friendly (asinh q/p) target, SAME winning net/settings (fair A/B vs the line above) ---
    ("qop/set-5L-524k", "config/detector/stereo2_qop.yaml",
     {"set-regressor": {"features": [[256, 256], [256, 256], [256, 128], [256, 128], [128, 64]]}}, 64, 50, 524288, 4.0e-4),
    # --- R11 (ABANDONED, user-killed): data-vs-duration control. Partial result settled it: at a FIXED
    #     ~327k steps, momR2 = 0.348 (32k) < 0.514 (131k) < 0.582 (524k=stereo2/set-16x) -> unique data
    #     is a real lever at matched compute (small pool overfits: 32k@640ep barely beats 32k@60ep). ---
    # --- R12: HIERARCHICAL set regressors on the layer-wise grid (does explicit nesting beat flat?).
    #     Dense per-straw aggregation (32x316 elements/event) is heavy -> batch 16, smaller pool. The
    #     fair baseline is layerwise/set (R1 = -0.17) on the SAME layer-wise representation. ---
    ("layerwise/set-65k", "config/detector/stereo_layerwise.yaml",
     {"set-regressor": {"features": [[128, 64], [128, 64]]}}, 64, 40, 65536, 5.0e-4),
    ("layerwise/double", "config/detector/stereo_layerwise.yaml",
     {"double-set-regressor": {"features": [[128, 64], [128, 64]], "global_dim": 64}}, 16, 40, 65536, 5.0e-4),
    ("layerwise/structured", "config/detector/stereo_layerwise.yaml",
     {"structured-set-regressor": {"features": [[128, 64], [128, 64]], "global_dim": 64}}, 16, 40, 65536, 5.0e-4),
]


def optimizer(lr=1.0e-3):
    return {"adamw": {"learning_rate": {"cosine_decay_schedule": {"init_value": float(lr)}}, "weight_decay": 1.0e-6}}


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_results(res):
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)
    write_leaderboard(res)


def agg(final):
    p = [final[k] for k in ("p_x", "p_y", "p_z") if k in final]
    v = [final[k] for k in ("vertex_x", "vertex_y", "vertex_z") if k in final]
    mom = sum(p) / len(p) if p else float("nan")
    vtx = sum(v) / len(v) if v else float("nan")
    return mom, vtx


def write_leaderboard(res):
    rows = [(n, r) for n, r in res.items() if r.get("status") == "ok"]
    rows.sort(key=lambda nr: agg(nr[1]["final"])[0])  # by momentum MSE (lower = better)
    lines = ["# Gap-study leaderboard (sorted by momentum MSE; R2 ~ 1-MSE)", "",
             "| rank | candidate | mom MSE | mom R2 | vtx MSE | p_x | p_y | p_z | vtx_z | min/cand |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (n, r) in enumerate(rows, 1):
        f = r["final"]
        mom, vtx = agg(f)
        lines.append(f"| {i} | {n} | {mom:.4f} | {1-mom:.3f} | {vtx:.4f} | {f.get('p_x',0):.3f} | "
                     f"{f.get('p_y',0):.3f} | {f.get('p_z',0):.3f} | {f.get('vertex_z',0):.3f} | {r.get('minutes',0):.1f} |")
    failed = [n for n, r in res.items() if r.get("status") != "ok"]
    if failed:
        lines += ["", "**failed/pending:** " + ", ".join(failed)]
    os.makedirs(OUT, exist_ok=True)
    with open(LEADER, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_one(name, det_path, reg_cfg, batch, epochs, samples, lr=1.0e-3):
    config = {
        "detector": yaml.safe_load(open(det_path)),
        "design": yaml.safe_load(open(DESIGN)),
        "regressor": reg_cfg,
        "optimizer": optimizer(lr),
        "training": {"batch": batch, "epochs": epochs, "samples": samples},
        # validation peak memory = one eval chunk; for the O(M^2) continuous-conv a 128-event chunk
        # OOMs, so cap the eval batch at the (smaller) training batch.
        "validation": {"batch": min(128, batch), "samples": VAL_SAMPLES},
        "sampling": {"batch": 1024},
    }
    ckpt = os.path.join(OUT, name.replace("/", "_"))
    t0 = time.time()
    history = REG.regress(seed=0, checkpoint=ckpt, restore=False, progress=False, **config)
    final = history[-1][2]
    return {"status": "ok", "final": {k: float(v) for k, v in final.items()},
            "minutes": (time.time() - t0) / 60.0, "epochs": epochs, "samples": samples, "batch": batch,
            "lr": lr, "history_tail": [[e, float(tr)] for e, tr, _ in history[-5:]]}


def main(only=None):
    res = load_results()
    for cand in CANDIDATES:
        name, det_path, reg_cfg, batch, epochs = cand[:5]
        samples = cand[5] if len(cand) > 5 else SAMPLES  # 6th element = pool size override (R3+)
        lr = cand[6] if len(cand) > 6 else 1.0e-3  # 7th element = peak LR override (R7+)
        if only is not None and name != only:
            continue
        if res.get(name, {}).get("status") == "ok":
            print(f"[skip] {name} (done)")
            continue
        print(f"[run ] {name} ...", flush=True)
        try:
            res[name] = run_one(name, det_path, reg_cfg, batch, epochs, samples, lr)
            print(f"[done] {name}: mom MSE {agg(res[name]['final'])[0]:.4f} in {res[name]['minutes']:.1f} min", flush=True)
        except Exception as e:
            res[name] = {"status": "error", "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
            print(f"[FAIL] {name}: {e}", flush=True)
        save_results(res)
    print("gap_study done.")


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    main(only=kw.get("only"))
