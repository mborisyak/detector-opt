from builtins import range as _range

__all__ = [
  'number_of_batches',
  'range'
]

def get_size(*arrays, axis: int):
  if any(x.ndim < axis + 1 for x in arrays):
    raise ValueError(f'Some arrays have lower dimensionality than implies by axis {axis}.')

  sizes = set([x.shape[axis] for x in arrays])

  if len(sizes) > 1:
    raise ValueError(f'Shapes of the inputs arrays are inconsistent across axis {axis}.')

  size, = sizes
  return size

def number_of_batches(size: int, batch: int, subbatch: bool=True):
  if subbatch:
    return size // batch + (0 if size % batch == 0  else 1)
  else:
    return size // batch

class range(object):
  def __init__(self, size: int, batch: int, subbatch: bool=True):
    self.size = size
    self.batch = batch

    self.n_full_batches = self.size // batch
    self.remainder = self.size % batch
    self.subbatch = subbatch

  def __len__(self):
    if self.subbatch and self.remainder > 0:
      return self.n_full_batches + 1
    else:
      return self.n_full_batches

  def __iter__(self):
    for i in _range(self.n_full_batches):
      index = slice(i * self.batch, (i + 1) * self.batch)
      yield index

    if self.remainder > 0 and self.subbatch:
      index = slice(self.size - self.remainder, self.size)
      yield index

# def batched_map(f, first, *rest, batch: int, axis: int=0):
#   arrays = (first, *rest)
#   size = get_size(*arrays, axis=axis)
#
#   result = jax.tree.map(
#     lambda shaped: jnp.zeros(shape=shaped.shape, dtype=shaped.dtype),
#     jax.eval_shape(f, first, *rest)
#   )
#
#   for index in batched_index(size, batch, axis=axis):
#     args = tuple(x[index] for x in arrays)
#     result_batch = f(*args)
#     result = jax.tree.map(
#       lambda acc, b: acc.at[index].set(b),
#       result, result_batch
#     )
#
#   return result