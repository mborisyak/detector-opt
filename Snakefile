# Snakemake driver for the multi-seed four-strategy BO comparison -- the DAG twin of make.sh (which
# stays; both drive the same scripts and the same output files, so they can be mixed freely). The
# point of the snakemake version is PARALLELISM: the seed x strategy jobs are independent processes
# (each builds its own detector, so the detector's thread-unsafety is irrelevant across jobs), and
# snakemake schedules as many as the machine allows.
#
#   snakemake -cN                     -> whatever is not yet computed, up to N jobs in parallel:
#                                        per seed 4 BO runs -> 4 verifications -> comparison.txt +
#                                        convergence_all.png, then the cross-seed median
#   snakemake -cN --resources gpu=K   -> at most K of the GPU-heavy jobs at once (each declares
#                                        `gpu=1`); pick K from GPU memory / per-job footprint
#   snakemake -c1                     -> strictly serial (a small card that fits one job)
#   snakemake -cN -n                  -> dry run: show what would be (re)computed
#
# Sharing one GPU between concurrent jobs requires XLA preallocation to be off -- with it on, the
# first job grabs the whole card and every other one OOMs. Every command below is prefixed with
# ENV (XLA_PYTHON_CLIENT_PREALLOCATE=false), so a job is correct regardless of the environment
# snakemake happened to inherit.
#
# Edit the CONFIG/SEEDS constants below; every path derives from them (a fresh comparison over new
# settings = new SEEDS or a new PREFIX).
#
# Recomputation is snakemake-native -- no --force is ever passed to the scripts. Snakemake deletes
# a job's declared outputs before running it, so a forced bo job starts clean (bo.py restarts when
# results.json is gone):
#
#   snakemake -cN --forcerun bo   -> redo the BO runs (and everything downstream)
#   snakemake -cN -F              -> redo absolutely everything
#
# The one file snakemake never deletes is verification.json (undeclared, see below):
# verify_trajectory.py reuses its points whenever the settings still match, so a forced verify only
# recomputes what a config change actually invalidated; delete the file by hand for a true
# from-scratch verification.
#
# verification.json is deliberately NOT a declared output: it is verify_trajectory.py's RESUMABLE
# state (verified points accumulate into it), and snakemake deletes the declared outputs of a failed
# or interrupted job. The completion-marker plots are declared instead, so a killed verification
# resumes for free on the next invocation. Config files are likewise not declared as inputs: a
# config touch must not silently schedule an hours-long BO rerun (a scheduled bo job deletes
# results.json first) -- force explicitly when settings change.
#
# ADOPTING OUTPUTS PRODUCED BY make.sh: files snakemake did not create have no provenance metadata,
# and its default rerun triggers (code/params, not just mtime) may schedule spurious reruns. Before
# the first real invocation over an existing output tree, either record metadata once with
# `snakemake -c1 --touch` or run with `--rerun-triggers mtime`; a `-n` dry run shows what would
# happen.
import random

CONFIG = "enzyme"  # run config: config/<CONFIG>.yaml
SUPER_SEED = 123456
rng = random.Random(SUPER_SEED)
SEEDS = [rng.randint(0, 2 ** 31 - 1) for _ in range(5)]

PREFIX = f"output/{CONFIG}"  # per-seed run tree = <PREFIX>/<seed>/<strategy>/, cross-seed median = <PREFIX>/median.png
STRATEGIES = ["from_scratch", "continue", "closest", "meta"]

# Prepended to every command below, so a job carries its environment explicitly rather than
# inheriting one: preallocation off (concurrent jobs must take only the GPU memory they actually
# use) and unbuffered stdout (progress is usually watched through a redirect or a pipe, where
# Python would block-buffer it and a healthy run would look silent for many minutes).
ENV = "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1"

wildcard_constraints:
  seed="|".join(str(seed) for seed in SEEDS),
  strategy="|".join(STRATEGIES),


rule all:
  input:
    expand(f"{PREFIX}/{{seed}}/comparison.txt", seed=SEEDS),
    f"{PREFIX}/median.png",


# One BO run: one config, one seed, one strategy, its own output tree.
rule bo:
  output:
    f"{PREFIX}/{{seed}}/{{strategy}}/results.json",
  resources:
    gpu=1,
  shell:
    f'{ENV} python scripts/bo.py "={CONFIG}" output={PREFIX}/{{wildcards.seed}}/{{wildcards.strategy}} '
    f"seed={{wildcards.seed}} nn_init_strategy={{wildcards.strategy}}"


# Independent re-scoring of the finished trajectory (scripts/verify_trajectory.py; see make.sh for
# what is and is not verified).
rule verify:
  input:
    f"{PREFIX}/{{seed}}/{{strategy}}/results.json",
  output:
    f"{PREFIX}/{{seed}}/{{strategy}}/verification.png",
    f"{PREFIX}/{{seed}}/{{strategy}}/verification_comparison.png",
  resources:
    gpu=1,
  shell:
    f'{ENV} python scripts/verify_trajectory.py "={CONFIG}" trajectory={PREFIX}/{{wildcards.seed}}/{{wildcards.strategy}} '
    f"seed={{wildcards.seed}}"


# Per-seed overlay of self-evaluated (dashed) vs verified (solid) + the comparison.txt table.
rule compare:
  input:
    expand(f"{PREFIX}/{{{{seed}}}}/{{strategy}}/results.json", strategy=STRATEGIES),
    expand(f"{PREFIX}/{{{{seed}}}}/{{strategy}}/verification.png", strategy=STRATEGIES),
  output:
    f"{PREFIX}/{{seed}}/comparison.txt",
    f"{PREFIX}/{{seed}}/convergence_all.png",
    f"{PREFIX}/{{seed}}/convergence_all.json",
  shell:
    f"{ENV} python scripts/compare_strategies.py --output {PREFIX}/{{wildcards.seed}}"


# Across seeds: per strategy the pointwise MEDIAN of the best-so-far (cummin) curves -- two panels,
# self-evaluated and verified (scripts/median_convergence.py).
rule median:
  input:
    expand(f"{PREFIX}/{{seed}}/{{strategy}}/results.json", seed=SEEDS, strategy=STRATEGIES),
    expand(f"{PREFIX}/{{seed}}/{{strategy}}/verification.png", seed=SEEDS, strategy=STRATEGIES),
  output:
    f"{PREFIX}/median.png",
    f"{PREFIX}/median.json",
  params:
    runs=" ".join(f"{PREFIX}/{seed}" for seed in SEEDS),
  shell:
    f"{ENV} python scripts/median_convergence.py --runs {{params.runs}} --output {PREFIX}/median.png"


# Stage targets mirroring make.sh's `bo` / `verify` stages (without their implied --force).
rule bo_all:
  input:
    expand(f"{PREFIX}/{{seed}}/{{strategy}}/results.json", seed=SEEDS, strategy=STRATEGIES),


rule verify_all:
  input:
    expand(f"{PREFIX}/{{seed}}/{{strategy}}/verification.png", seed=SEEDS, strategy=STRATEGIES),
