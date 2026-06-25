from collections import namedtuple
import numpy as np

import jax
import jax.numpy as jnp

from tqdm import tqdm

Point = namedtuple('Point', ['x', 'y', 'z'])

def main():
  ### checks the proper way to fill a jax buffer
  seed = 123
  rng = np.random.default_rng(seed)

  n, b, m = int(5 * 1024 // 3), 1024 // 4, 1024

  buffer = Point(
    x=jnp.zeros(shape=(n * b, m), dtype=jnp.float32),
    y=jnp.zeros(shape=(n * b, m), dtype=jnp.float32),
    z=jnp.zeros(shape=(n * b, m), dtype=jnp.float32),
  )
  buffer_np = Point(
    x=np.zeros(shape=(n * b, m), dtype=np.float32),
    y=np.zeros(shape=(n * b, m), dtype=np.float32),
    z=np.zeros(shape=(n * b, m), dtype=np.float32),
  )

  counter = jnp.zeros(shape=(), dtype=jnp.int32)

  def assign(buff, c, values):
    size, = set(s.shape[0] for s in jax.tree.leaves(values))

    index = c + jnp.arange(size, dtype=jnp.int32)
    updated = jax.tree.map(
      lambda b, v: b.at[index].set(v),
      buff, values
    )

    return updated, c + size

  assign = jax.jit(assign, donate_argnums=(0, ))

  def assign2(buff, index, values):
    updated = jax.tree.map(
      lambda b, v: b.at[index].set(v),
      buff, values
    )

    return updated

  assign2 = jax.jit(assign2, donate_argnums=(0, ))

  for i in tqdm(range(n)):
    indx = np.arange(i * b, (i + 1) * b)
    vs = Point(rng.normal(size=(b, m)), rng.normal(size=(b, m)), rng.normal(size=(b, m)))

    buffer, counter = assign(buffer, counter, vs)
    buffer_np.x[indx] = vs.x
    buffer_np.y[indx] = vs.y
    buffer_np.z[indx] = vs.z

  for i in tqdm(range(n)):
    indx = np.arange(i * b, (i + 1) * b)
    assert jnp.all(buffer.x[indx] == buffer_np.x[indx])
    assert jnp.all(buffer.y[indx] == buffer_np.y[indx])
    assert jnp.all(buffer.z[indx] == buffer_np.z[indx])

  for i in tqdm(range(n)):
    indx = np.arange(i * b, (i + 1) * b)
    vs = Point(rng.normal(size=(b, m)), rng.normal(size=(b, m)), rng.normal(size=(b, m)))
    buffer = assign2(buffer, indx, vs)

if __name__ == '__main__':
  main()