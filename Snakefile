# Snakemake driver for the multi-seed multi-strategy BO comparison -- the DAG twin of make.sh (which
# stays; both drive the same scripts and the same output files, so they can be mixed freely). The
# point of the snakemake version is PARALLELISM: the seed x strategy jobs are independent processes
# (each builds its own detector, so the detector's thread-unsafety is irrelevant across jobs), and
# snakemake schedules as many as the machine allows.
#
# THE TASK IS INFERRED FROM THE PATH, which is `<prefix>/<config>/...`. `<prefix>` is any path,
# however many directories deep; `<config>` is exactly ONE directory and names the run config
# `config/<config>.yaml`. So asking for a file names both where the campaign goes and what it is:
#
#   snakemake -c4 output/linear/median.json              -> the debug task, start to finish
#   snakemake -c4 output/enzyme_extremes/median.json     -> the extremes campaign
#   snakemake -c4 output/rehearsal/2026-08/linear/median.json   -> the same task, somewhere else
#   snakemake -cN                                        -> DEFAULT_PREFIX/DEFAULT_TASK (`rule all`)
#
# The split is unambiguous because <config> is pinned to the alternation of configs that exist: in
# `output/rehearsal/2026-08/linear/median.json` only `linear` can be the config.
#
# Nothing else needs editing to add a task: drop a run config in config/ and ask for its median.
#
#   snakemake -cN                     -> whatever is not yet computed, up to N jobs in parallel:
#                                        per seed one BO run per strategy -> verifications ->
#                                        comparison.txt + convergence_all.png, then the cross-seed
#                                        median
#   snakemake -cN --resources local_gpu=K
#                                     -> at most K of the GPU-heavy jobs at once (each declares
#                                        `local_gpu=1`); pick K from GPU memory / per-job footprint
#   snakemake -c1                     -> strictly serial (a small card that fits one job)
#   snakemake -cN -n                  -> dry run: show what would be (re)computed
#
# SEED SCHEDULING: no ordering -- every seed x strategy BO job is independent, so snakemake runs as
# many at once as you allow. Cap concurrency with --resources local_gpu=K (each heavy rule declares
# local_gpu=1, and 0 for a CPU-only task, which therefore is not throttled by that counter):
#
#   snakemake -c16 --resources local_gpu=8
#                                     -> up to 8 GPU jobs at once. The H100 has ample room (one job
#                                        ~15% of its 132 SMs, ~1.2 GB); pick K from the 4-vs-8
#                                        saturation probe. preallocation-off is baked into ENV below.
#
# Sharing one GPU between concurrent jobs requires XLA preallocation to be off -- with it on, the
# first job grabs the whole card and every other one OOMs. Every command below is prefixed with
# ENV (XLA_PYTHON_CLIENT_PREALLOCATE=false), so a job is correct regardless of the environment
# snakemake happened to inherit.
#
# Recomputation is snakemake-native -- no --force is ever passed to the scripts. Snakemake deletes
# a job's declared outputs before running it, so a forced bo job starts clean:
#
#   snakemake -cN --forcerun bo   -> redo the BO runs (and everything downstream)
#   snakemake -cN -F              -> redo absolutely everything
#
# A KILLED `bo` JOB IS RESUMED, NOT REDONE, and it needs nothing from you. `bo.py` writes its
# trajectory to `partial.json` while a run is in flight and only writes `results.json` -- this rule's
# declared output -- when the budget pool has filled. A killed run therefore leaves the output
# ABSENT: snakemake reschedules the rule on the next invocation, and bo.py picks up from
# `partial.json` plus the state it commits between designs (`optimizer.npz` + `trainer.npz`), which
# are deliberately undeclared so snakemake never removes them. No `--rerun-incomplete`, no stale
# completion flag, and no half-written trajectory ever appears under the name a consumer asks for.
#
# ⚠️ A FORCED `bo` DOES throw the run away -- but only a FINISHED one. `--forcerun bo` deletes
# results.json, and with no partial.json beside it bo.py starts fresh and overwrites the state on its
# first commit; that costs the paid-for event pool, not just the loop position. Forcing a run that
# was KILLED does nothing, because its partial.json is undeclared and survives, and bo.py resumes
# from it. Delete `partial.json` by hand to make such a run start over.
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
#
# RUNNING UNDER SLURM. This machine schedules everything through SLURM: one GPU exposed as TWO
# shards on partition `main`, so at most two GPU jobs run at once and the rest queue. Launch with
#
#   snakemake --executor slurm --default-resources slurm_partition=main -j8
#
# and let SLURM, not snakemake, do the queueing -- `-j` only bounds how many jobs snakemake has in
# flight, while the shard count bounds how many actually run. The heavy rules (`bo`, `verify`)
# therefore declare SLURM resources rather than relying on the local `local_gpu=K` counter:
#
#   cpus_per_task     what a run actually uses
#   mem_mb            memory is cgroup-ENFORCED here (ConstrainRAMSpace=yes) and the site default is
#                     only DefMemPerCPU=2000, i.e. 8000 MB at 4 CPUs; an under-request is an
#                     OOM-kill many hours in. 12000 still lets two run at once against the node's
#                     26000 MB
#   runtime=2880      minutes. A BO run here is of order 5 h and the partition itself is
#                     time-unlimited; the value exists only so nothing is killed by a wall clock
#   gres="shard:1"    shard, NOT gpu:1, which would take the card exclusively and collide with the
#                     other shard job. EMPTY for a task whose run config declares `device: cpu` --
#                     the plugin tests the resource for truthiness, so "" submits no --gres at all
#                     and the job runs beside the GPU work instead of queueing behind a shard it does
#                     not need
#
# All four are per-task, taken from the run config's own `device` (see `device()` below).
#
# TWO SPELLINGS THAT DO NOT WORK, both verified against snakemake-executor-plugin-slurm 2.8.0:
#
#   slurm_extra="'--gres=shard:1'" is REJECTED outright -- the plugin reserves --gres for itself
#   (validation.py) and fails submission. The native `gres` resource above is the supported route.
#
#   a `gpu` resource must NOT be set alongside it. set_gres_string() appends `--gpus=<n>` whenever
#   `gpu` is present, so `gpu=1` would submit `--gres=shard:1 --gpus=1`, and the --gpus half claims
#   the whole card exclusively -- the precise collision the shard split exists to prevent. The local
#   concurrency counter is therefore renamed `local_gpu`, a name the plugin does not interpret; cap
#   a non-SLURM run with `--resources local_gpu=K` exactly as `gpu=K` used to.
#
# ENV omits the MPS pipe directory: these finish-up runs are on a local RTX 3070, where MPS measured
# neutral-to-negative for enzyme.
import glob
import os
import random

# Every run config under config/ is a task. Leading-underscore files are templates and rehearsals,
# not campaigns, so they are excluded -- which also keeps the wildcard alternation short.
TASKS = sorted(
  os.path.splitext(os.path.basename(path))[0] for path in glob.glob("config/*.yaml")
  if not os.path.basename(path).startswith("_")
)
# What `snakemake -cN` with no target builds. Override on the command line: --config task=linear.
DEFAULT_TASK = config.get("task", "enzyme_extremes")
# Only for the no-target invocation; every explicit target carries its own prefix in the path.
DEFAULT_PREFIX = config.get("prefix", "output")

# WHERE A TASK RUNS IS THE RUN CONFIG'S OWN `device`, read here rather than restated. A second list
# in this file could disagree with it, and the failure would be silent in the worse direction: a
# `device: cpu` task queueing behind a GPU shard it never uses, or a `device: cuda` task submitted
# without one and falling back to the host mid-campaign.
def device(task):
  import yaml

  with open(f"config/{task}.yaml") as handle:
    return str((yaml.safe_load(handle) or {}).get("device", "cuda")).strip().lower()


CPU_ONLY = {task for task in TASKS if device(task) == "cpu"}

SUPER_SEED = 123456
rng = random.Random(SUPER_SEED)
SEEDS = [rng.randint(0, 2 ** 31 - 1) for _ in range(2)]

STRATEGIES = ["from_scratch", "meta"]

# Prepended to every command below, so a job carries its environment explicitly rather than
# inheriting one: preallocation off (concurrent jobs must take only the GPU memory they actually
# use), unbuffered stdout (progress is usually watched through a redirect or a pipe, where Python
# would block-buffer it and a healthy run would look silent for many minutes), and the CUDA MPS pipe
# dir so every job routes through the user-mode MPS server. WITHOUT MPS, N processes TIME-SLICE the
# H100 (measured ~0.20 SMACT total regardless of N -> ~no speedup); WITH it they co-run (measured
# ~0.69 SMACT / 365 W at 8 jobs -> ~5x aggregate throughput). The daemon must be started once per
# boot (user-mode, no root):
#   CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps nvidia-cuda-mps-control -d     (stop: echo quit | nvidia-cuda-mps-control)
ENV = "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1"

wildcard_constraints:
  # `prefix` is ANY path and spans directories (snakemake wildcards match `/` by default); `task` is
  # exactly ONE directory, pinned to the alternation of existing run configs, which is what makes the
  # split unambiguous -- in `output/a/b/linear/median.json` only `linear` can be the task.
  prefix=".+",
  task="|".join(TASKS),
  seed="|".join(str(seed) for seed in SEEDS),
  strategy="|".join(STRATEGIES),


def gres(wildcards):
  return "" if wildcards.task in CPU_ONLY else "shard:1"


def local_gpu(wildcards):
  return 0 if wildcards.task in CPU_ONLY else 1


def cpus_per_task(wildcards):
  return 2 if wildcards.task in CPU_ONLY else 4


def mem_mb(wildcards):
  return 3000 if wildcards.task in CPU_ONLY else 12000


rule all:
  input:
    expand(f"{DEFAULT_PREFIX}/{DEFAULT_TASK}/{{seed}}/comparison.txt", seed=SEEDS),
    f"{DEFAULT_PREFIX}/{DEFAULT_TASK}/median.png",


# One BO run: one task, one seed, one strategy, its own output tree. No inputs -- every seed x
# strategy job is independent, scheduled freely up to the shard count (or --resources local_gpu=K).
rule bo:
  output:
    "{prefix}/{task}/{seed}/{strategy}/results.json",
  resources:
    local_gpu=local_gpu,
    cpus_per_task=cpus_per_task,
    mem_mb=mem_mb,
    runtime=2880,
    gres=gres,
  shell:
    f"{ENV} python scripts/bo.py \"={{wildcards.task}}\" "
    "output={wildcards.prefix}/{wildcards.task}/{wildcards.seed}/{wildcards.strategy} "
    "seed={wildcards.seed} nn_init_strategy={wildcards.strategy}"


# Independent re-scoring of the finished trajectory (scripts/verify_trajectory.py; see make.sh for
# what is and is not verified).
rule verify:
  input:
    "{prefix}/{task}/{seed}/{strategy}/results.json",
  output:
    "{prefix}/{task}/{seed}/{strategy}/verification.png",
    "{prefix}/{task}/{seed}/{strategy}/verification_comparison.png",
  resources:
    local_gpu=local_gpu,
    cpus_per_task=cpus_per_task,
    mem_mb=mem_mb,
    runtime=2880,
    gres=gres,
  shell:
    f"{ENV} python scripts/verify_trajectory.py \"={{wildcards.task}}\" "
    "trajectory={wildcards.prefix}/{wildcards.task}/{wildcards.seed}/{wildcards.strategy} "
    "seed={wildcards.seed} progress=plain"


# Per-seed overlay of self-evaluated (dashed) vs verified (solid) + the comparison.txt table.
rule compare:
  input:
    expand("{{prefix}}/{{task}}/{{seed}}/{strategy}/results.json", strategy=STRATEGIES),
    expand("{{prefix}}/{{task}}/{{seed}}/{strategy}/verification.png", strategy=STRATEGIES),
  output:
    "{prefix}/{task}/{seed}/comparison.txt",
    "{prefix}/{task}/{seed}/convergence_all.png",
    "{prefix}/{task}/{seed}/convergence_all.json",
  shell:
    f"{ENV} python scripts/compare_strategies.py "
    "--output {wildcards.prefix}/{wildcards.task}/{wildcards.seed}"


# Across seeds: per strategy the pointwise MEDIAN of the best-so-far (cummin) curves -- two panels,
# self-evaluated and verified (scripts/median_convergence.py).
rule median:
  input:
    expand("{{prefix}}/{{task}}/{seed}/{strategy}/results.json", seed=SEEDS, strategy=STRATEGIES),
    expand("{{prefix}}/{{task}}/{seed}/{strategy}/verification.png", seed=SEEDS, strategy=STRATEGIES),
  output:
    "{prefix}/{task}/median.png",
    "{prefix}/{task}/median.json",
  params:
    runs=lambda wildcards: " ".join(f"{wildcards.prefix}/{wildcards.task}/{seed}" for seed in SEEDS),
  shell:
    f"{ENV} python scripts/median_convergence.py --runs {{params.runs}} "
    "--output {wildcards.prefix}/{wildcards.task}/median.png"


# Stage targets mirroring make.sh's `bo` / `verify` stages (without their implied --force). They take
# the task from --config task=<name>, since a rule with no output cannot carry a wildcard.
rule bo_all:
  input:
    expand(f"{DEFAULT_PREFIX}/{DEFAULT_TASK}/{{seed}}/{{strategy}}/results.json", seed=SEEDS, strategy=STRATEGIES),


rule verify_all:
  input:
    expand(f"{DEFAULT_PREFIX}/{DEFAULT_TASK}/{{seed}}/{{strategy}}/verification.png", seed=SEEDS, strategy=STRATEGIES),
