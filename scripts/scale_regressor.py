"""Fresh-pool resampling trainer: push a (data-bound) regressor toward its ceiling with UNBOUNDED data
while the GPU sample buffer stays at a fixed size. Each cycle resamples a fresh pool (new seed) and
RESUMES the model+optimizer from the checkpoint -- so over N cycles the net sees N x pool unique events.
Reuses scripts/regression.py::regress (resume + fresh pool come for free); uses a CONSTANT LR so resume
across cycles is clean. Records per-cycle held-out per-component MSE (R2 ~ 1-MSE).

Run:  python scripts/scale_regressor.py            # default = stereo2 3-layer set, the gap-study winner
See tasks/fairship-gap-study.md (R6)."""
import os
import sys
import json
import time

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import regression as REG  # noqa: E402

DESIGN = "config/design/initial_stereo.yaml"


def momentum_mse(final):
    return sum(final[k] for k in ("p_x", "p_y", "p_z")) / 3.0


def vertex_mse(final):
    return sum(final[k] for k in ("vertex_x", "vertex_y", "vertex_z")) / 3.0


def scale(detector_cfg, regressor_cfg, checkpoint, *, pool=262144, batch=64, epochs_per_cycle=8,
          cycles=24, lr=3.0e-4, val_samples=8192, log_path=None):
    base = {
        "detector": yaml.safe_load(open(detector_cfg)),
        "design": yaml.safe_load(open(DESIGN)),
        "regressor": regressor_cfg,
        "optimizer": {"adamw": {"learning_rate": float(lr), "weight_decay": 1.0e-6}},  # CONSTANT lr
        "training": {"batch": batch, "epochs": epochs_per_cycle, "samples": pool},
        "validation": {"batch": min(128, batch), "samples": val_samples},
        "sampling": {"batch": 1024},
    }
    log = []
    t0 = time.time()
    for c in range(cycles):
        history = REG.regress(seed=1000 + c, checkpoint=checkpoint, restore=(c > 0), progress=False, **base)
        final = {k: float(v) for k, v in history[-1][2].items()}
        seen = (c + 1) * pool
        row = {"cycle": c, "events_seen": seen, "mom_mse": momentum_mse(final),
               "mom_r2": 1 - momentum_mse(final), "vtx_r2": 1 - vertex_mse(final),
               "minutes": (time.time() - t0) / 60.0, "final": final}
        log.append(row)
        print(f"[cycle {c:2d}] seen={seen/1e6:.2f}M  momR2={row['mom_r2']:+.3f}  vtxR2={row['vtx_r2']:+.3f}  "
              f"p_z={final['p_z']:.3f}  ({row['minutes']:.0f} min)", flush=True)
        if log_path:
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
    return log


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = kw.get("checkpoint", "data/gap_study/scale_stereo2_big")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    scale(
        detector_cfg=kw.get("detector", "config/detector/stereo2.yaml"),
        regressor_cfg={"set-regressor": {"features": [[256, 128], [256, 128], [128, 64]]}},
        checkpoint=out,
        pool=int(kw.get("pool", 262144)),
        batch=int(kw.get("batch", 64)),
        epochs_per_cycle=int(kw.get("epochs_per_cycle", 8)),
        cycles=int(kw.get("cycles", 24)),
        lr=float(kw.get("lr", 3.0e-4)),
        log_path=out + "_log.json",
    )
