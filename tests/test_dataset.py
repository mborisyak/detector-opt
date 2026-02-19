import numpy as np
import detopt

def test_dataset(seed):
  rng = np.random.default_rng(seed)
  dataset = detopt.utils.dataset.Dataset(10, (), ())

  for i in range(10):
    x = np.arange(i * 3, (i + 1) * 3)
    dataset.add(x, x)
    x_sample, _ = dataset.sample(rng, 1000)
    x_sample = x_sample.astype(int)
    print(np.bincount(x_sample, minlength=33))
    print()