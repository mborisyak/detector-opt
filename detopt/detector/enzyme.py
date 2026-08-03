import jax
import jax.numpy as jnp

### A + B -> C + D via E

ZERO_CELSIUS =  273.15
REFERENCE_TEMPERATURE = 10.0 + ZERO_CELSIUS
INV_TEMPERATURE_SPAN = 1 / ZERO_CELSIUS - 1 / REFERENCE_TEMPERATURE

__all__ = [
  'kinetics'
]

def vant_hoff(T, log_K_0, Q10):
  """
  K_0 --- rate at 0C;
  Q100 - K_ref / K_0, K_ref - rate at the reference temperature, 100C;
  """
  delta = 1 / ZERO_CELSIUS - 1 / (T + ZERO_CELSIUS)
  return jnp.exp(
    log_K_0 + jnp.log(Q10) * delta / INV_TEMPERATURE_SPAN
  )

def gibbs_fraction(temperature, delta_H, delta_C, temperature_melting):
  """
  delta_H --- entalpy of unfolding;
  delta_G --- heat capacity change;
  delta H and delta G are assumed to be normalized by R.
  """
  T = temperature + ZERO_CELSIUS
  T_m = temperature_melting + ZERO_CELSIUS

  dG = delta_H * (1 - T / T_m) - \
      delta_C * ((T_m - T) - T * jnp.log(T_m / T))

  return jax.nn.sigmoid(dG / T)

def kinetics(A, B, C, D, E, temperature, parameters):
  K_A = vant_hoff(temperature, parameters['log_K0_A'], parameters['Q10_A'])
  K_B = vant_hoff(temperature, parameters['log_K0_B'], parameters['Q10_B'])
  Ki_C = vant_hoff(temperature, parameters['log_K0i_C'], parameters['Q10_C'])
  Ki_D = vant_hoff(temperature, parameters['log_K0i_D'], parameters['Q10_D'])

  Kapp_A = K_A * (1 + C / Ki_C)
  Kapp_B = K_B * (1 + D / Ki_D)

  ### actually Arrhenius, but the expression is the same
  k_cat = vant_hoff(temperature, parameters['log_k0_cat'], parameters['Q10_cat'])

  active_enzyme = gibbs_fraction(temperature, parameters['delta_H'], parameters['delta_C'], parameters['T_melting'])

  E_a = active_enzyme * E

  rate = k_cat * E_a * A * B / (A + Kapp_A) / (B + Kapp_B)

  return rate
