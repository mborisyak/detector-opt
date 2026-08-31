#!/usr/bin/env python3
"""Score ONE middle-of-BO design under ONE retention setting -- stage 2 of `docs/final.md`.

    python scripts/probe_retention.py =angle strategy=angle-meta-rewind-025 \
        trajectory=output/final/angle/validation/282522616/meta/results.json \
        seed=282522616 output=output/final/angle/probe/282522616/meta/rewind-025.json

WHAT IS BEING SELECTED, AND WHY IT NEEDS A PROBE RATHER THAN A CAMPAIGN. The retention setting
(`rewind`, or shrink-and-perturb) only acts at a DATA ADDITION, so its effect is a property of one
design's training, not of a whole BO trajectory. Scoring it on a full run would spend a campaign's
budget to measure a per-design quantity and would confound it with every downstream proposal the
setting happened to shift. This scores the SAME design under each setting and compares the losses.

THE STATE THE DESIGN STARTS FROM IS THE VALIDATION RUN'S OWN, not a reconstruction:

  * `trainer.replay(rows[:k])` rebuilds both event pools by RE-SIMULATING the committed trajectory.
    `detector(design, event_index)` is deterministic and the index is a function of (detector size,
    seed, generations), so the pools come back exactly as the validation run had them -- this is the
    same call `bo.py` makes to resume, not a probe-specific approximation.
  * For `meta` that same call runs `_replay_carried_state`, which reads the carried network back from
    the checkpoint the validation run wrote at design k-1. That IS "regenerate context for meta":
    the replay buffer is the replayed pool and the carried network is the checkpoint.
  * For `continue`, `restore_design_parameters(k - 1)` loads the preceding design's trained network,
    which is what `bo.py` hands the arm mid-campaign.
  * For `from_scratch` there is nothing to load, and the requirement is instead that the FRESH draw
    match: `train` derives it from the per-iteration seed, and the per-iteration seed is the k-th
    spawn of the run's iteration branch. Spawning k times here puts this probe on exactly the stream
    the validation run was on, so the initial network is the same draw bit for bit.

⚠️ THE CHECKPOINT DIRECTORY IS COPIED, NEVER SHARED. `train` writes a checkpoint at convergence, so
pointing this at the validation run's own directory would have seven retention settings overwriting
the trajectory they are all reading from. The copy is small (one network per design) and it keeps the
validation run byte-identical after any number of probes.

THE DESIGN IS THE POSITIONAL MIDDLE of the trajectory, `len(rows) // 2`, so every setting scores the
same design and the arms differ in which design they get -- which is what `docs/final.md` asks for
("1 per strategy-seed"), since a trajectory's designs are its own and `meta` banks more of them.
"""
import json
import os
import shutil

import numpy as np

import detopt.detector

from bo import TRAINERS, drop_foreign_knobs, _resolve_strategy


def probe_retention(output, trajectory, seed: int, design_index: int = -1, **config):
  config = _resolve_strategy(config)
  arm = config["nn_init_strategy"]
  config = drop_foreign_knobs(config, arm)

  with open(trajectory) as handle:
    rows = json.load(handle)["results"]
  if len(rows) < 3:
    raise SystemExit(f"probe_retention: {trajectory} holds {len(rows)} designs; need at least 3")
  index = len(rows) // 2 if design_index < 0 else int(design_index)
  if not 0 < index < len(rows):
    raise SystemExit(f"probe_retention: design index {index} outside 1..{len(rows) - 1}")

  source_checkpoints = os.path.join(os.path.dirname(trajectory), "checkpoints")
  if not os.path.isdir(source_checkpoints):
    raise SystemExit(f"probe_retention: no checkpoints beside {trajectory}; the arm cannot be warm-started")
  work = os.path.splitext(output)[0] + ".checkpoints"
  if os.path.isdir(work):
    shutil.rmtree(work)
  shutil.copytree(source_checkpoints, work)

  detector = detopt.detector.from_config(config["detector"])
  network_seq, iteration_seq = np.random.SeedSequence(int(seed)).spawn(2)
  trainer = TRAINERS.get(arm, TRAINERS["per_design"]).from_config(
    detector, config, checkpoint_dir=work, seed=int(network_seq.generate_state(1)[0]),
  )
  trainer.replay(rows[:index])

  # The iteration branch is consumed one spawn per completed design, so the k-th design's seed is the
  # (k+1)-th spawn. Spawning k first is what `bo.py`'s own resume does.
  iteration_seq.spawn(index)
  iteration_seed = int(iteration_seq.spawn(1)[0].generate_state(1)[0])

  init_params = None
  if arm in ("continue", "closest"):
    if arm == "continue":
      donor = index - 1
    else:
      history = np.asarray([row["x_scaled"] for row in rows[:index]], dtype=np.float64)
      donor = int(np.argmin(np.linalg.norm(history - np.asarray(rows[index]["x_scaled"], np.float64)[None, :], axis=1)))
    print(f"[warm-start] {arm}: design {donor}", flush=True)
    init_params = trainer.restore_design_parameters(donor)

  design_scaled = np.asarray(rows[index]["x_scaled"], dtype=np.float32)
  print(f"[probe] {arm} @ design {index}/{len(rows)} of {trajectory}", flush=True)

  # THE WHOLE CONVERGENCE CURVE IS THE MEASUREMENT, not just its endpoint. Stage 3 ranks the settings
  # POINTWISE along a best-so-far trajectory, so one probed design contributes as many comparisons as
  # it has epochs instead of one. `_snapshot` is CUMULATIVE -- each call carries the full history so
  # far -- and `on_epoch` is dispatched to a worker thread, so completions can arrive out of order;
  # keeping the LONGEST snapshot seen is what makes this independent of that ordering.
  longest = {}

  def on_epoch(snapshot, _longest=longest):
    if len(snapshot["train_loss_per_epoch"]) >= len(_longest.get("train_loss_per_epoch", ())):
      _longest.clear()
      _longest.update(snapshot)

  result = trainer.train(design_scaled, iteration_seed, init_params=init_params, on_epoch=on_epoch, step=index)

  # THE SAME QUANTITY THE CAMPAIGN RECORDS. `bo.py` scores a design as `objective_loss +
  # design_penalty`, and on `intersection_penalty` that second term is not zero. It cannot change
  # which retention setting wins -- the design is fixed across the seven settings, so the penalty is
  # a shared constant -- but recording only the trained loss would leave `loss` silently offset from
  # the `reference_loss` copied out of the trajectory, which is how a spurious discrepancy gets read
  # as a bug in the probe.
  design_physical = detector.to_nominal(design_scaled)
  penalty = detector.design_penalty(design_physical)
  penalty = None if penalty is None else float(penalty)
  trained_loss = float(result.objective_loss)

  record = {
    "arm":
    arm,
    "seed":
    int(seed),
    "trajectory":
    trajectory,
    "design_index":
    index,
    "n_designs":
    len(rows),
    "x_scaled": [float(v) for v in design_scaled],
    "rewind":
    float(config["training"].get("rewind", 0.0)),
    "shrink":
    float(config["training"].get("shrink", 1.0)),
    "param_noise":
    float(config["training"].get("param_noise", 0.0)),
    "loss":
    trained_loss if penalty is None else trained_loss + penalty,
    "trained_loss":
    trained_loss,
    "design_penalty":
    penalty,
    "loss_std":
    float(result.objective_std),
    # The per-epoch curve stage 3 consumes. `objective` is `(train + val) / 2` epoch by epoch -- the
    # SAME quantity `TrainResult.objective_loss` reports at convergence, so a best-so-far built from
    # it is on the scale the campaign actually scores. `window` is the training rows behind each
    # epoch, which is the x-axis the settings are interpolated onto: it is directly measured, it is
    # monotone, and it is what a setting that converges cheaply spends less of.
    "objective_per_epoch":
    [round(float(a + b) / 2.0, 8) for a, b in zip(longest["train_loss_per_epoch"], longest["val_loss_per_epoch"])],
    "train_per_epoch": [round(float(v), 8) for v in longest["train_loss_per_epoch"]],
    "val_per_epoch": [round(float(v), 8) for v in longest["val_loss_per_epoch"]],
    "window_per_epoch": [int(v) for v in longest["train_budget_per_epoch"]],
    # The per-design ceiling on training rows, AFTER the trainer rounds it down onto the ladder. It
    # is the x where stage 3 places each trajectory's pseudo point, and it is identical across the
    # settings, so every curve ends up spanning the same range.
    "iteration_limit":
    int(trainer.iteration_limit),
    "spent":
    int(result.spent),
    "reference_loss":
    rows[index].get("loss"),
  }
  os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
  with open(output, "w") as handle:
    json.dump(record, handle, indent=2)
  shutil.rmtree(work, ignore_errors=True)
  print(f"[probe] loss {record['loss']:.6f}+-{record['loss_std']:.4f} on {record['spent']} calls -> {output}", flush=True)
  return record


if __name__ == "__main__":
  import sys

  import gearup

  gearup.gearup(probe_retention).with_config("config/root.yaml")(sys.argv[1:])
