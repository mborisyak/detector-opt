#!/usr/bin/env python3
"""Does an interrupted BO run come back as the SAME run? -- an end-to-end check of the resume protocol.

    srun --cpus-per-task=4 -u python scripts/check_resume.py --config linear \
        --output output/debug/resume --kill-after 2

WHAT IS CHECKED, and why a weaker check would not do. ``scripts/bo.py`` persists the optimiser's
evidence and the trainer's event pools between designs and restarts an interrupted run at the design
boundary. The claim that makes that worth having is not "it starts again" -- it is that the run
continues on the SAME TRAJECTORY: the same proposals, the same losses, the same budget spend as if
nothing had happened. So this script runs the identical configuration twice,

* REFERENCE -- start to finish, uninterrupted;
* INTERRUPTED -- the same seed into a separate directory, killed with SIGKILL once ``--kill-after``
  designs have been scored (no cleanup, no chance to flush), then restarted and left to finish,

and compares the two row by row from the kill point onward. A driver that lost its seed position, its
Sobol block or a slice of its event pool would diverge exactly there, and nowhere earlier.

SIGKILL rather than SIGTERM deliberately: the state on disk must be sufficient by itself, not
sufficient given a graceful shutdown path that a node failure would never run.

THE COMPARISON IS EXACT for the per-design strategies (``from_scratch`` / ``continue`` / ``closest``):
every draw is seeded from the run's own sequence, so the resumed run reproduces the reference to the
bit. For ``meta`` it is NOT, and the script says so rather than pretending: the continual network's
Adam moments are deliberately not persisted, so the design after a restart carries fresh moments and
its loss moves by a small amount. That arm is reported with its differences rather than asserted to
be zero.
"""

import argparse
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time


REPOSITORY = str(pathlib.Path(__file__).resolve().parent.parent)


def trajectory(directory):
    """The path a run's trajectory is at right now: ``results.json`` once it has finished, otherwise
    the ``partial.json`` it writes while in flight. Their names are the completion flag."""
    finished = os.path.join(directory, "results.json")
    return finished if os.path.exists(finished) else os.path.join(directory, "partial.json")


def complete_rows(directory):
    """The scored rows of a run's trajectory, or ``[]`` if there is none yet or it is half-written."""
    path = trajectory(directory)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return [r for r in json.load(f).get("results", []) if r.get("status", "complete") == "complete"]
    except (json.JSONDecodeError, KeyError):
        return []  # caught mid-write; the next poll sees it whole


def launch(config, output, seed, extra):
    return subprocess.Popen(
        [sys.executable, "scripts/bo.py", f"={config}", f"output={output}", f"seed={seed}", *extra],
        cwd=REPOSITORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_to_completion(config, output, seed, extra, label):
    started = time.time()
    process = launch(config, output, seed, extra)
    out, _ = process.communicate()
    print(f"[{label}] exit {process.returncode} after {time.time() - started:.0f} s, "
          f"{len(complete_rows(output))} designs")
    if process.returncode != 0:
        print(out[-4000:])
        raise SystemExit(f"{label} failed")
    return out


def run_until_killed(config, output, seed, extra, kill_after, poll, timeout):
    """Start the run and SIGKILL it once ``kill_after`` designs are on the record."""
    process = launch(config, output, seed, extra)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"the run finished ({process.returncode}) before {kill_after} designs -- "
                             f"lower --kill-after or raise the budget")
        if len(complete_rows(output)) >= kill_after:
            # The row is written BEFORE the state pair is committed, so the kill may land in that
            # window -- which is the interesting case, not a flaw in the check: the resume drops the
            # uncommitted row and re-measures that design, and the comparison below still has to come
            # out identical. Kill hard: the state on disk must stand on its own.
            os.kill(process.pid, signal.SIGKILL)
            process.wait()
            return len(complete_rows(output))
        time.sleep(poll)
    os.kill(process.pid, signal.SIGKILL)
    raise SystemExit(f"timed out waiting for {kill_after} designs")


def compare(reference, resumed, first, tolerance):
    """Row-by-row differences from ``first`` onward. Returns the largest absolute loss/design gap."""
    if len(reference) != len(resumed):
        print(f"  LENGTH DIFFERS: reference {len(reference)} designs, resumed {len(resumed)}")
    worst_loss, worst_design = 0.0, 0.0
    print(f"  {'iter':>4}  {'reference':>10}  {'resumed':>10}  {'|dloss|':>9}  {'|ddesign|':>9}  spent")
    for index in range(min(len(reference), len(resumed))):
        a, b = reference[index], resumed[index]
        d_loss = abs(float(a["loss"]) - float(b["loss"]))
        d_design = max(abs(float(p) - float(q)) for p, q in zip(a["design"], b["design"]))
        marker = "" if index < first else ("  ok" if d_loss <= tolerance and d_design <= tolerance else "  DIFFERS")
        if index >= first:
            worst_loss, worst_design = max(worst_loss, d_loss), max(worst_design, d_design)
        print(f"  {index:>4}  {float(a['loss']):>10.6f}  {float(b['loss']):>10.6f}  "
              f"{d_loss:>9.2e}  {d_design:>9.2e}  {a['spent']}/{b['spent']}{marker}")
    return worst_loss, worst_design


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="linear", help="run config name, as bo.py's `=<name>`")
    parser.add_argument("--output", default="output/debug/resume", help="directory for both runs")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kill-after", type=int, default=2, help="designs to score before the SIGKILL")
    parser.add_argument("--strategy", default="from_scratch", help="nn_init_strategy for both runs")
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="allowed |difference| after the restart; 0 demands bit-identical, which is "
                             "what the per-design strategies must deliver")
    parser.add_argument("--poll", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--set", nargs="*", default=[], help="extra `key=value` overrides for bo.py")
    arguments = parser.parse_args()

    reference_dir = os.path.join(arguments.output, "reference")
    resumed_dir = os.path.join(arguments.output, "interrupted")
    for path in (reference_dir, resumed_dir):
        shutil.rmtree(path, ignore_errors=True)
    extra = [f"nn_init_strategy={arguments.strategy}", *arguments.set]

    print(f"REFERENCE: an uninterrupted run of `{arguments.config}` at seed {arguments.seed} "
          f"({arguments.strategy})", flush=True)
    run_to_completion(arguments.config, reference_dir, arguments.seed, extra, "reference")

    print(f"INTERRUPTED: the same run, SIGKILLed after {arguments.kill_after} designs", flush=True)
    killed_at = run_until_killed(arguments.config, resumed_dir, arguments.seed, extra,
                                 arguments.kill_after, arguments.poll, arguments.timeout)
    print(f"  killed with {killed_at} designs on the record", flush=True)
    # A killed run must leave `partial.json` and NO `results.json` -- that is what keeps a build
    # system from ever seeing a half-written trajectory under the name it declared as an output.
    for name in ("partial.json", "optimizer.npz", "trainer.npz"):
        path = os.path.join(resumed_dir, name)
        print(f"  {name}: {os.path.getsize(path)} bytes" if os.path.exists(path) else f"  {name}: MISSING")
    if os.path.exists(os.path.join(resumed_dir, "results.json")):
        print("  results.json: PRESENT -- a killed run must not have written it")

    print("RESUMING", flush=True)
    output = run_to_completion(arguments.config, resumed_dir, arguments.seed, extra, "resumed")
    resume_lines = [line for line in output.splitlines() if line.startswith("[resume]")]
    print("\n".join(f"  {line}" for line in resume_lines) if len(resume_lines) > 0
          else "  NO [resume] LINE -- the run restarted from scratch instead of resuming")

    reference = complete_rows(reference_dir)
    resumed = complete_rows(resumed_dir)
    print(f"\nCOMPARISON from iteration {killed_at} (the first design the resumed run produced itself):")
    worst_loss, worst_design = compare(reference, resumed, killed_at, arguments.tolerance)
    print(f"\nworst |dloss| = {worst_loss:.3e}, worst |ddesign| = {worst_design:.3e}, "
          f"tolerance {arguments.tolerance:.3e}")
    exact = worst_loss <= arguments.tolerance and worst_design <= arguments.tolerance
    same_length = len(reference) == len(resumed)
    print("RESUME REPRODUCES THE REFERENCE" if exact and same_length else
          "RESUME DIVERGES FROM THE REFERENCE" + ("" if same_length else " (and ran a different number of designs)"))
    raise SystemExit(0 if exact and same_length else 1)


if __name__ == "__main__":
    main()
