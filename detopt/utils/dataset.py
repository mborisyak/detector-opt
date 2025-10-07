import numpy as np

__all__ = [
  'Dataset'
]

class Dataset(object):
  def __init__(self, capacity, condition_shape, sample_shape, target_shape):
    self.capacity = capacity
    self.condition_shape = condition_shape
    self.sample_shape = sample_shape
    self.target_shape = target_shape

    self.C = np.ndarray(shape=(capacity, *condition_shape), dtype=np.float32)
    self.X = np.ndarray(shape=(capacity, *sample_shape), dtype=np.float32)
    self.Y = np.ndarray(shape=(capacity, *target_shape), dtype=np.float32)

    self.offset = 0
    self.size = 0

  def add(self, c, x, y):
    n, *_ = c.shape
    assert x.shape[0] == n
    assert y.shape[0] == n

    indx = np.arange(self.offset, self.offset + n) % self.capacity
    self.C[indx], self.X[indx], self.Y[indx] = c, x, y

    self.size = min(self.capacity, self.size + n)
    self.offset = (self.offset + n) % self.capacity

  def sample(self, rng: np.random.Generator, n):
    indx = rng.integers(0, self.size, size=(n, ))
    return self.C[indx], self.X[indx], self.Y[indx]