"""Why a GROUPED convolution's backward pass fails on the CERN GPU nodes but not on the workstation.

Reports the CUDA/cuDNN stack the process actually loaded, then runs four convolution trials at the
shapes `StripRegressor` uses, so a failure can be attributed to the op, to grouping, or to the
environment. Prints one line per trial and never raises: a trial that dies reports its exception.

The trials, in order of what each one rules out:

* dense       -- a NON-grouped conv backward. If this fails the CUDA stack is broken outright.
* grouped     -- the same through ``nnx.Conv`` with ``feature_group_count=C``, i.e. the failing op.
* grouped_lax -- the same through ``jax.lax.conv_general_dilated`` directly, with the dimension
                 numbers flax computes. Distinguishes a flax wrapper problem from an XLA/cuDNN one;
                 they should behave identically.
* grouped_nograph -- grouped again with command buffers off, isolating CUDA-graph capture from the
                 convolution itself.

RUN WITH NO XLA_FLAGS SET. The point is to observe the DEFAULT behaviour -- autotuning on, command
buffers on -- because the flags that suppress these failures also cost algorithm selection and CUDA
graphs, which is most of JAX's advantage on this model.
"""

import glob
import os
import subprocess
import sys

B, POSITIONS, CHANNELS, KERNEL, STRIDE = 32, 316, 32, 9, 5


def report_environment():
  print(f"[env] XLA_FLAGS      = {os.environ.get('XLA_FLAGS')!r}")
  print(f"[env] LD_LIBRARY_PATH= {os.environ.get('LD_LIBRARY_PATH', '')[:240]}")
  smi = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap,memory.total', '--format=csv,noheader'],
                       capture_output=True, text=True)
  print(f"[env] nvidia-smi     = {smi.stdout.strip() or smi.stderr.strip()[:160]}")
  import jax
  import jaxlib
  print(f"[env] jax {jax.__version__}  jaxlib {jaxlib.__version__}  python {sys.version.split()[0]}")
  device = jax.devices()[0]
  print(f"[env] device         = {device} kind={device.device_kind} cc={getattr(device, 'compute_capability', '?')}")
  for root in [p for p in sys.path if p.endswith('site-packages')]:
    found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, 'nvidia', '*')))
    if len(found) > 0:
      print(f"[env] nvidia wheels  = {root.split('/')[-3]}: {' '.join(found)}")
  for name in ('libcudnn.so', 'libcublas.so', 'libcudart.so'):
    hits = [p for p in glob.glob(f"/proc/{os.getpid()}/maps")]
    if len(hits) > 0:
      with open(hits[0]) as handle:
        loaded = sorted({line.split()[-1] for line in handle if name.split('.so')[0] in line})
      print(f"[env] loaded {name:<14}= {loaded[:2] if len(loaded) > 0 else 'NOT LOADED'}")


def trial(name, build):
  """Run one convolution's forward+backward and report, converting any failure into a printed line."""
  import jax
  import jax.numpy as jnp
  try:
    step = build()
    value = float(step())
    print(f"[trial] {name:<16} OK        grad-norm {value:.4e}")
    return True
  except Exception as error:  # noqa: BLE001 -- the whole point is to classify the failure
    text = str(error).replace('\n', ' ')
    print(f"[trial] {name:<16} FAILED    {type(error).__name__}: {text[:260]}")
    return False


def _flax_conv(groups):
  import jax
  import jax.numpy as jnp
  import optax
  from flax import nnx
  conv = nnx.Conv(CHANNELS, CHANNELS, kernel_size=(KERNEL, ), strides=(STRIDE, ), padding='SAME',
                  feature_group_count=groups, rngs=nnx.Rngs(0))
  graphdef, params = nnx.graphdef(conv), nnx.state(conv, nnx.Param)
  x = jax.random.normal(jax.random.key(0), (B, POSITIONS, CHANNELS))

  @jax.jit
  def step(params):
    grads = jax.grad(lambda p: jnp.sum(nnx.merge(graphdef, p)(x)**2))(params)
    return optax.global_norm(grads)

  return lambda: step(params)


def _lax_conv():
  """The same grouped convolution through ``jax.lax`` with flax's own dimension numbers."""
  import jax
  import jax.numpy as jnp
  from jax import lax
  kernel = jax.random.normal(jax.random.key(1), (KERNEL, 1, CHANNELS)) / KERNEL**0.5
  x = jax.random.normal(jax.random.key(0), (B, POSITIONS, CHANNELS))
  dimension_numbers = lax.ConvDimensionNumbers((0, 2, 1), (2, 1, 0), (0, 2, 1))

  @jax.jit
  def step(kernel):
    def loss(k):
      y = lax.conv_general_dilated(x, k, (STRIDE, ), 'SAME', dimension_numbers=dimension_numbers,
                                   feature_group_count=CHANNELS)
      return jnp.sum(y**2)

    return jnp.linalg.norm(jax.grad(loss)(kernel))

  return lambda: step(kernel)


def main():
  report_environment()
  print(f"[shapes] batch {B} strip {POSITIONS} channels {CHANNELS} kernel {KERNEL} stride {STRIDE}")
  trial('dense', lambda: _flax_conv(1))
  trial('grouped', lambda: _flax_conv(CHANNELS))
  trial('grouped_lax', _lax_conv)


if __name__ == '__main__':
  main()
