import jax
import jax.numpy as jnp
import jax.nn as jnn


def cross_attention(xs, inducing, scale_features, *, mask_xs=None):
  # xs: (*, n, t)
  # is: (*, m, t)
  # K: (*, n, m, t)
  K = jax.nn.sigmoid(
    scale_features * (xs[..., :, None, :] - inducing[..., None, :, :])
  ) * mask_xs[..., None, None]

  inducing_total = jnp.sum(K, axis=-2) + 1
  xs_total = jnp.sum(K, axis=-3) + 1

  xs_aggregated = jnp.sum(K * inducing[..., None, :, :], axis=-2) / inducing_total
  inducing_aggregated = jnp.sum(K * xs[..., None, :], axis=-3) / xs_total

  return xs_aggregated, inducing_aggregated

def test_dual_attention(seed):
    rng = jax.random.PRNGKey(seed)
    rng, key_xs, key_inducing, key_mask, key_scale = jax.random.split(rng, num=5)

    n_b, n, m, k = 1024, 17, 11, 7
    n_ext = 5

    ### (b, n, k)
    xs = jax.random.normal(key_xs, shape=(n_b, n, k))
    ### (b, m, k)
    inducing = 10 * jax.random.normal(key_inducing, shape=(n_b, m, k))
    scale = jax.random.normal(key_scale, shape=(k, ))

    mask = jax.random.bernoulli(key_mask, shape=(n_b, n))

    x1, inducing1 = cross_attention(xs, inducing, scale, mask_xs=mask)

    print('x', jnp.std(xs), jnp.std(x1))
    print('inducing', jnp.std(inducing), jnp.std(inducing1))

    print(x1.shape, inducing1.shape)

    assert x1.shape == xs.shape
    assert inducing1.shape == inducing.shape

    rng, key_xs_extension = jax.random.split(rng, num=2)
    xs_ext = jax.random.normal(key_xs_extension, shape=(n_b, n_ext, k))
    xs_extended = jnp.concatenate([xs, xs_ext], axis=-2)
    mask_extended = jnp.concatenate([mask, jnp.zeros(shape=(n_b, n_ext), dtype=mask.dtype)], axis=-1)

    x2, inducing2 = cross_attention(xs_extended, inducing, scale, mask_xs=mask_extended)

    assert x2.shape == xs_extended.shape
    assert inducing2.shape == inducing.shape

    assert jnp.allclose(x2[:, :n], x1)
    assert jnp.allclose(inducing1, inducing2)