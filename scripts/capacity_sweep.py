#!/usr/bin/env python3
"""Does the network's CAPACITY set the converged loss, at a design held fixed?

    python scripts/capacity_sweep.py --n-designs 3 --output output/capacity/sweep.json

The stopping rule exits a design when both loss curves are plateaued and train agrees with val. An
under-powered network satisfies both at its own ceiling, so the reported loss would be biased by
capacity rather than set by the design's information content -- and nothing in the criterion would
notice. `scripts/capacity_probe.py` showed the symptom: on the SAME designs `meta` reports
+0.0103 (median, 4/4 repeats) above `from_scratch` while spending a third less. This measures the
cause directly: the SAME design, trained under configurations of differing capacity, comparing the
converged LEVEL (not the gap, and not the cost of reaching it).

The grid separates the four things the shipped config confounds. It runs `set-regressor` at
`[[24, 16], [16, 24]]` with `n_models: 4` AND `p_dropout: 0.1`, i.e. it ensembles and regularises a
small network at the same time:

  * width      -- [[24,16],[16,24]] against [[64,48],[48,64]]: raw capacity of the shipped stack.
  * depth      -- `alpha-set-regressor`: the same deep set with each per-element block replaced by a
                  residual stack whose per-unit `alpha` starts at ZERO, so the block is the identity
                  at initialisation and depth costs nothing in conditioning. This is the peer
                  session's architecture, built here for this detector.
  * ensemble   -- n_models 4 against 1: averaging members, each of the same width.
  * dropout    -- 0.1 against 0: note this is ACTIVATION dropout (`nnx.Dropout` between layers),
                  not drop-connect -- no mask touches `EnsembleLinear`'s kernel -- so at a block
                  width of 16-24 it removes whole features, which is a large perturbation.

Read the LEVEL column. If a higher-capacity network converges lower on the same design, "converged"
means "hit its own ceiling" and every loss this benchmark reports is capacity-limited. If the level
is flat across the grid, capacity is not the binding constraint and the exit rule is reporting what
the design can actually support.

Two things to know before reading an alpha-resnet row. Its residual branches are inert at step zero
(alpha = 0), so a dropout mask inside a branch cannot change the output there -- a peer session saw
its dropout column come out bit-identical on the first design for exactly this reason, which is an
artefact of the initialisation and not evidence that dropout does nothing. And that peer measured
the analogous grid on a DIFFERENT detector (`enzyme_profile`): ensemble-vs-dropout moved the exit
window by up to 1.86x while the level moved only +-1.2% with no consistent sign -- but with the
ensemble OFF, designs never exited at all, the gap staying open past 100k calls. That is the
opposite signature from `meta` here (fewer calls, SMALLER gap), which is the reason to measure
capacity on THIS detector rather than assume it transfers.
"""
import argparse
import copy
import json
import os

import numpy as np

import detopt.detector
import detopt.utils.config
from detopt.nn.trainer import DesignTrainer

# (label, regressor config). Each entry REPLACES config["regressor"] wholesale, so an entry names
# its architecture as well as its hyper-parameters -- the grid crosses architectures, and
# `set-regressor`'s `features` (a list of blocks) and `alpha-set-regressor`'s (a list of block
# OUTPUT widths, with the stack described by width/depth) are not the same knob.
NARROW = [[24, 16], [16, 24]]
WIDE = [[64, 48], [48, 64]]
VARIANTS = (
    ("set     narrow  n=4 drop=0.1", {"set-regressor": {"features": NARROW, "n_models": 4, "p_dropout": 0.1}}),
    ("set     narrow  n=4 drop=0  ", {"set-regressor": {"features": NARROW, "n_models": 4, "p_dropout": 0.0}}),
    ("set     narrow  n=1 drop=0  ", {"set-regressor": {"features": NARROW, "n_models": 1, "p_dropout": 0.0}}),
    ("set     WIDE    n=4 drop=0.1", {"set-regressor": {"features": WIDE, "n_models": 4, "p_dropout": 0.1}}),
    ("alpha   w64 d4  n=4 drop=0.1",
     {"alpha-set-regressor": {"features": [16, 24], "width": 64, "depth": 4, "n_models": 4, "p_dropout": 0.1}}),
    ("alpha   w64 d4  n=4 drop=0  ",
     {"alpha-set-regressor": {"features": [16, 24], "width": 64, "depth": 4, "n_models": 4, "p_dropout": 0.0}}),
    ("alpha   w64 d4  n=1 drop=0  ",
     {"alpha-set-regressor": {"features": [16, 24], "width": 64, "depth": 4, "n_models": 1, "p_dropout": 0.0}}),
    ("alpha   w32 d2  n=4 drop=0.1",
     {"alpha-set-regressor": {"features": [16, 24], "width": 32, "depth": 2, "n_models": 4, "p_dropout": 0.1}}),
)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="enzyme")
  parser.add_argument("--n-designs", type=int, default=3, help="fixed designs, each seen by every variant")
  parser.add_argument("--designs-from", default=None,
                      help="results.json to take the designs from, at --quantiles of its reported "
                           "loss. Uniformly random designs are a bad probe of CAPACITY: in the "
                           "[25, 80] box most of them are uninformative, the level sits at the 1/3 "
                           "no-information ceiling for every variant, and the comparison measures "
                           "nothing (measured: a random design scored 0.3701 and cost 91456 calls). "
                           "Capacity can only bind where there is signal to fit.")
  parser.add_argument("--quantiles", type=float, nargs="*", default=[0.0, 0.4, 0.8],
                      help="quantiles of the run's reported loss to sample designs at (0 = best)")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--only", default=None, help="comma-separated substrings; run just those variants")
  parser.add_argument("--iteration-limit", type=int, default=None,
                      help="override training.iteration_limit. It is the EPOCH LENGTH "
                           "(iteration_limit // batch SGD steps between convergence checks), not just "
                           "a per-design cap, so this is the arm that asks whether the coarse check "
                           "inflates the spend -- a peer measured 2048 steps/epoch still growing past "
                           "window 61440 where 512 steps/epoch exited at 12288.")
  parser.add_argument("--output", required=True)
  arguments = parser.parse_args()

  base = detopt.utils.config.load_config(f"config/{arguments.config}.yaml")
  if arguments.iteration_limit is not None:
    base["training"]["iteration_limit"] = arguments.iteration_limit
    print(f"iteration_limit -> {arguments.iteration_limit} "
          f"({arguments.iteration_limit // int(base['training']['batch'])} steps/epoch)", flush=True)
  detector = detopt.detector.from_config(
      detopt.utils.config.load_config(f"config/detector/{base['detector']}.yaml"))
  if arguments.designs_from is None:
    rng = np.random.default_rng(arguments.seed)
    designs = rng.random((arguments.n_designs, int(detector.design_dim()))).astype(np.float32)
    labels = [f"random {i}" for i in range(len(designs))]
  else:
    with open(arguments.designs_from) as f:
      results = json.load(f)["results"]
    order = sorted(results, key=lambda r: r["loss"])
    picked = [order[min(len(order) - 1, int(round(q * (len(order) - 1))))] for q in arguments.quantiles]
    designs = np.asarray([p["x_scaled"] for p in picked], np.float32)
    labels = [f"q{q:.1f} reported {p['loss']:.4f}" for q, p in zip(arguments.quantiles, picked)]
  print("designs: " + "; ".join(labels), flush=True)

  variants = VARIANTS
  if arguments.only is not None:
    wanted = arguments.only.split(",")
    variants = tuple(v for v in VARIANTS if any(w in v[0] for w in wanted))

  rows = []
  for index, design in enumerate(designs):
    print(f"\n=== design {index} ({labels[index]}) ===", flush=True)
    for label, regressor in variants:
      config = copy.deepcopy(base)
      config["regressor"] = copy.deepcopy(regressor)
      trainer = DesignTrainer.from_config(detector, config, checkpoint_dir=None,
                                          seed=arguments.seed + index)
      result = trainer.train(design, np.random.SeedSequence(arguments.seed + index), step=0)
      row = {"design": index, "variant": label.strip(), "regressor": regressor,
             "loss": None if result is None else float(result.objective_loss),
             "std": None if result is None else float(result.objective_std),
             "spent": None if result is None else int(result.spent)}
      rows.append(row)
      with open(arguments.output, "w") as f:  # written as it goes: a kill costs the run in flight only
        json.dump({"rows": rows}, f, indent=2)
      if result is not None:
        print(f"  {label}  level {row['loss']:.4f}  gap+err {row['std']:.4f}  "
              f"{row['spent']:7d} calls", flush=True)
      else:
        print(f"  {label}  DID NOT CONVERGE within the pool", flush=True)

  os.makedirs(os.path.dirname(arguments.output) or ".", exist_ok=True)
  with open(arguments.output, "w") as f:
    json.dump({"rows": rows}, f, indent=2)

  print("\nMEDIAN OVER DESIGNS (the level is the question; the spend is context)")
  print(f"{'variant':30s} {'level':>8s} {'gap+err':>8s} {'calls':>9s} {'n':>3s}")
  for label, _ in variants:
    got = [r for r in rows if r["variant"] == label.strip() and r["loss"] is not None]
    if len(got) == 0:
      print(f"{label:30s} {'-- never converged --':>28s}")
      continue
    print(f"{label:30s} {np.median([r['loss'] for r in got]):8.4f} "
          f"{np.median([r['std'] for r in got]):8.4f} {np.median([r['spent'] for r in got]):9.0f} "
          f"{len(got):3d}")


if __name__ == "__main__":
  main()
