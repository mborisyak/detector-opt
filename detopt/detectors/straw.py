import jax
import jax.numpy as jnp

__all__ = [
  'solve'
]

def boris_step(x, v, q_inv_m_B, dt):
  # Magnetic field rotation
  t = 0.5 * dt * q_inv_m_B
  t_norm_sqr = jnp.sum(jnp.square(t), axis=-1)
  v_ = v + jnp.cross(v, t)
  v_updated = v + 2 * jnp.cross(v_, t) / (1 + t_norm_sqr)[..., None]
  x_updated = x + v_updated * dt

  return x_updated, v_updated

def solve(x0, v0, q, m, J, K, L, *, B0: jax.Array | None=None, dt: float, steps: int):
  A = jnp.array([
    [J, -K, 0],
    [K, -J, 0],
    [0, 0, 0]
  ], dtype=x0.dtype)
  # Bx = (J * x + K * y) exp(-z^2 / L^2)
  # By = (-K * x - J * y) exp(-z^2 / L^2)

  def step(state, _):
    x, v = state
    B_ = jnp.exp(-jnp.square(x[..., -1] / L))[..., None] * jnp.matmul(x, A)
    B = B_ if B0 is None else (B_ + B0)
    q_inv_m_B = B * (0.5 * q[..., None] / m[..., None])
    x_updated, v_updated = boris_step(x, v, q_inv_m_B, dt)

    state_updated = (x_updated, v_updated)
    return state_updated, x_updated
  # (n, n_t, 3)
  _, trajectory = jax.lax.scan(
    step,
    init=(x0, v0),
    xs=None,
    length=steps
  )

  return trajectory

  # (n, n_layers, n_tubes) 0.0 if no hit, 1.0 if hit

def project(
  trajectory: jax.Array,
  z0s: jax.Array, angles: jax.Array, lengths: jax.Array, straw_r: jax.Array,
  n_straws: int
):
  # trajectory: (n_t, n, 3)
  # z0s: (n_layers, )
  x, y, z = trajectory[..., 0], trajectory[..., 1], trajectory[..., 2]
  alpha = (z0s - z[:-1]) / (z[1:] - z[:-1])
  intersections = alpha * (trajectory[1:] - trajectory[:-1]) + trajectory[:-1]

  n_x, n_y = jnp.cos(angles), jnp.sin(angles)

  a = x * n_x + y * n_y
  delta_x = x - a * n_x
  delta_y = y - a * n_y
  delta = jnp.sqrt(jnp.square(delta_x) + jnp.square(delta_y))

  index = delta / (2 * straw_r)
  return


class StrawDetector(object):
  def __init__(
    self, layers, origin, momentum,
    J, K, L, dt,
    sigma_x0: float=1.0e-1, sigma_v0: float=1.0e-1,
    lambda_m: float=1.0
  ):
    """
    :param layers: layers in form [(z_0, angle_0, length_0, height_0), ...]
    :param origin:
    :param momentum:
    :param J:
    :param K:
    :param L:
    :param dt:
    :param sigma_x0:
    :param sigma_v0:
    """
    self.layers = layers

    self.origin = origin
    self.momentum = momentum

    self.sigma_x0 = sigma_x0
    self.sigma_v0 = sigma_v0
    self.lambda_m = lambda_m

    self.J = J
    self.K = K
    self.L = L
    self.dt = dt

    self.layers = layers
    self.max_L = max(z for z, _ in layers)

    min_v_z = self.momentum[-1] - 3 * self.sigma_v0

    self.steps = 2 * L / min_v_z / dt

  def __call__(self, rng: jax.Array, n: int):
    key_x0, key_v0, key_q, key_m = jax.random.split(rng, num=4)

    x0 = self.sigma_x0 * jax.random.normal(key_x0, shape=(n, 3)) + self.origin
    v0 = self.sigma_v0 * jax.random.normal(key_v0, shape=(n, 3)) + self.momentum

    q = 2.0 * jax.random.bernoulli(key_q, p=0.5, shape=(n, )) - 1.0
    m = self.lambda_m * jax.random.exponential(key_m, shape=(n, ))

    trajectory = solve(
      x0, v0, q=q, m=m,
      J=self.J, K=self.K, L=self.L,
      B0=None, dt=self.dt, steps=self.steps
    )

    return trajectory



