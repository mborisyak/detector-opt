import detopt

import math
import numpy as np
from scipy import special as sps
import jax
import jax.numpy as jnp

from flax import nnx

MAX_TDC = 2000

SQRT_2 = math.sqrt(2)
INV_SQRT_2 = math.sqrt(0.5)

def uniform_to_normal(xs):
  eps = np.finfo(xs.dtype).eps
  us = np.clip(xs, eps - 1, 1 - eps)

  return sps.erfinv(us) * SQRT_2

def normal_to_uniform(xs):
  us = sps.special.erf(xs * INV_SQRT_2)
  return us

def train(seed, dataset_path, **config):
  data = np.load(dataset_path)
  positions = data['positions']
  mask = data['mask']
  tdc = data['tdc']
  hnl_momenta = data['hnl_momenta']
  hnl_decay = data['hnl_decay']
  hnl_mass = data['hnl_mass']
  reconstructed_momenta = data['reconstructed_momenta']
  reconstructed = data['reconstructed']
  design = data['design']

  stations = config['stations']
  views = config['views']
  layers = config['layers']
  tubes = config['tubes']

  position_normalization = np.array([stations, views, layers, tubes], dtype=np.float32)
  momentum_normalization = np.array(config['normalization']['momentum'])
  decay_normalization = np.array(config['normalization']['decay'])
  tdc_normalization = np.array(config['normalization']['tdc'])
  design_lower = np.array(config['design']['lower'], dtype=np.float32)
  design_upper = np.array(config['design']['upper'], dtype=np.float32)
  design_center = (design_lower + design_upper) / 2
  design_delta = (design_upper - design_lower) / 2

  def encode_design(x):
    return uniform_to_normal((x - design_center) / design_delta)

  def decode_design(x):
    return normal_to_uniform(x) * design_delta + design_center

  def sample(sampling_rng: np.random.Generator, batch, ):
    index = sampling_rng.integers(low=0, high=positions.shape[0], size=(batch, ))

    X_batch = np.concatenate([positions[index] / position_normalization, tdc[index] / tdc_normalization], axis=-1)
    mask_batch = mask[index]
    y_batch = np.concatenate([
      hnl_decay[index] / decay_normalization,
      hnl_momenta[index] / momentum_normalization
    ], axis=-1)
    c_batch = encode_design(design[index])

    return X_batch, mask_batch, y_batch, c_batch

  rng = np.random.default_rng(seed)
  jax_rng = jax.random.PRNGKey(rng.integers(low=0, high=2 ** 31))
  nnx_rngs = nnx.Rngs(jax_rng)

  regressor = detopt.nn.SparseDeepSet(5, 6, depth=5, hidden=32, rngs=nnx_rngs)

  X, m, y, c = sample(rng, 12)

  p = regressor(X, m)
  print(p.shape, y.shape)

  import matplotlib.pyplot as plt

  fig = plt.figure(figsize=(24, 18))
  subfigs = fig.subfigures(1, 2)
  axes = subfigs[0].subplots(3, 2)
  for i in range(3):
    axes[i, 0].hist(hnl_momenta[:, i], bins=100, histtype='step')
  for i in range(3):
    axes[i, 1].hist(hnl_decay[:, i], bins=100, histtype='step')

  axes = subfigs[1].subplots(2, 1)
  axes[0].hist(np.minimum(tdc[mask].ravel(), 2000), bins=100, histtype='step')
  fig.tight_layout()
  fig.savefig('hists.png')
  plt.close(fig)

if __name__ == '__main__':
  import gearup

  gearup.gearup(
    train=train
  ).with_config('config/bo.yaml')()
