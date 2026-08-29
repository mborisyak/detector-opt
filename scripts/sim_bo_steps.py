"""Simulate BO on a GP ground truth fitted to the SHiP 1M corpus.

Fits one GP to all banked (design, loss) pairs, takes its posterior MEAN as a noiseless ground-truth
objective, and runs the repository's own `BayesianOptimizer` against it. Measures what +20% more BO
steps buys, and how noisy that gain is across independent runs.

The gain is measured over INDEPENDENT runs, never as a prefix of one run: best-so-far is a running
minimum, so within a single run `best@1.2n <= best@n` holds by construction and P(improve) is 1 for
free. The paired figure is reported alongside only to show that gap.
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp

from detopt.bo import BayesianOptimizer
from detopt.bo.jax_gp import fit_gp, gp_factorize, gp_predict

GP_CFG = dict(
  n_folds=5, n_restarts=5, n_steps=40, log_lengthscale_prior_bounds=[-6.0, 4.0], log_amplitude_prior_bounds=[-6.0, 1.5]
)
EI_CFG = dict(n_restarts=32, n_steps=100)


def build_ground_truth(path, seed=0, fit_restarts=4, fit_steps=80):
  rows = json.load(open(path))
  X = jnp.asarray([r["x"] for r in rows], dtype=jnp.float32)
  y = jnp.asarray([r["loss"] for r in rows], dtype=jnp.float32)
  s = np.asarray([r["loss_std"] if r["loss_std"] is not None else np.nan for r in rows], dtype=np.float32)
  s = np.where(np.isnan(s), np.nanmedian(s), s)
  noise = jnp.asarray(s, dtype=jnp.float32)
  y_mean = float(jnp.mean(y))
  state = fit_gp(
    jax.random.key(seed), X, y - y_mean, noise, n_restarts=8, n_steps=200, log_lengthscale_prior_bounds=(-6.0, 4.0),
    log_amplitude_prior_bounds=(-6.0, 1.5), n_folds=5
  )
  factors = gp_factorize(X, y - y_mean, state.hyper_parameters, noise)

  @jax.jit
  def truth(x):
    mean, _ = gp_predict(factors, jnp.atleast_2d(x))
    return mean[0] + y_mean

  return truth, state, float(np.median(s)), X, y


def one_run(truth, d, n_steps, seed, obs_noise, rng):
  bo = BayesianOptimizer(d, gp=GP_CFG, ei=EI_CFG, kernel=None, n_init=GP_CFG["n_folds"])
  xs, ys, trues = [], [], []
  for k in range(n_steps):
    x = np.asarray(bo.propose(int(seed * 100003 + k)), dtype=np.float32)
    f = float(truth(jnp.asarray(x)))
    y = f + float(rng.normal(0.0, obs_noise))
    bo.append(x, y, obs_noise)
    xs.append(x)
    ys.append(y)
    trues.append(f)
  ys = np.asarray(ys)
  trues = np.asarray(trues)
  best_obs = float(np.min(ys))
  true_at_recommended = float(trues[int(np.argmin(ys))])
  return best_obs, true_at_recommended, trues


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--designs", default="/tmp/ship1m_designs.json")
  ap.add_argument("--runs", type=int, default=200)
  ap.add_argument("--n", type=int, nargs="+", default=[10])
  ap.add_argument("--factor", type=float, default=1.2)
  ap.add_argument("--out", default="/tmp/sim_bo_steps.json")
  ap.add_argument("--fit-restarts", type=int, default=4)
  ap.add_argument("--fit-steps", type=int, default=80)
  args = ap.parse_args()

  t_fit = time.time()
  truth, state, obs_noise, X, y = build_ground_truth(args.designs, fit_restarts=args.fit_restarts, fit_steps=args.fit_steps)
  print(f"  oracle GP fitted on {X.shape[0]} designs in {time.time()-t_fit:.0f}s", flush=True)
  ls = np.exp(np.asarray(state.hyper_parameters.log_lengthscale))
  print(
    f"  ground-truth GP: lengthscales {np.round(ls, 3).tolist()}  "
    f"amplitude {float(np.exp(state.hyper_parameters.log_amplitude)):.4f}  obs noise {obs_noise:.5f}"
  )
  d = X.shape[1]
  out = {"obs_noise": obs_noise, "lengthscales": ls.tolist(), "results": []}

  for n in args.n:
    m = int(round(n * args.factor))
    t0 = time.time()
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(22)
    def batch(steps, base, rng, tag):
      acc = []
      for r in range(args.runs):
        acc.append(one_run(truth, d, steps, base + r, obs_noise, rng))
        if (r + 1) % 10 == 0:
          print(f"    {tag}: {r+1}/{args.runs} at {time.time()-t0:.0f}s", flush=True)
      return acc
    A = batch(n, 1000, rng_a, "A(n)")
    B = batch(m, 9000, rng_b, "B(m) independent")
    # nested/paired view: run A extended to m steps reuses A's own prefix
    P = batch(m, 1000, rng_a, "P(m) nested")

    for key, idx in [("true_at_recommended", 1), ("best_observed", 0)]:
      a = np.array([r[idx] for r in A])
      b = np.array([r[idx] for r in B])
      p = np.array([r[idx] for r in P])
      pairs = (b[None, :] < a[:, None]).mean()
      adv = a.mean() - b.mean()
      sem = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
      boot = np.array([
        np.random.default_rng(s).choice(a, len(a)).mean() - np.random.default_rng(s + 77).choice(b, len(b)).mean()
        for s in range(2000)
      ])
      out["results"].append(
        dict(
          metric=key, n=n, m=m, mean_n=float(a.mean()), mean_m=float(b.mean()), advantage=float(adv), sem=float(sem),
          p_better_independent=float(pairs), ci=[float(np.percentile(boot, 2.5)),
                                                 float(np.percentile(boot, 97.5))], sd_n=float(a.std(ddof=1)),
          sd_m=float(b.std(ddof=1)), p_better_nested=float((p < a).mean()), advantage_nested=float(a.mean() - p.mean())
        )
      )
    print(f"  n={n} -> m={m} done in {time.time()-t0:.0f}s")

  json.dump(out, open(args.out, "w"), indent=1)
  print()
  print(f"  {'metric':<22}{'n':>4}{'m':>4}{'loss@n':>9}{'loss@m':>9}{'gain':>9}{'+-':>8}{'P(m<n) indep':>14}{'P nested':>10}")
  for r in out["results"]:
    print(
      f"  {r['metric']:<22}{r['n']:>4}{r['m']:>4}{r['mean_n']:>9.4f}{r['mean_m']:>9.4f}"
      f"{r['advantage']:>9.4f}{r['sem']:>8.4f}{r['p_better_independent']:>14.3f}{r['p_better_nested']:>10.3f}"
    )
  print(f"  written {args.out}")


if __name__ == "__main__":
  main()
