import os
import yaml

import numpy as np

def main(seed, n, output, **config):
  rng = np.random.default_rng(seed)

  bounds = config['bounds']

  os.makedirs(output, exist_ok=True)

  low, high = bounds['view_angle']
  angles = np.linspace(low, high, num=n)

  for i in range(n):
    geometry = {}

    for k, bound in bounds.items():
      if k == 'view_angle':
        geometry[k] = float(angles[i])
      else:
        geometry[k] = bound

    configuration = {'SST': geometry}

    with open(os.path.join(output, f'strawtubes_{i}.yaml'), 'w') as f:
      yaml.dump(configuration, f)

if __name__ == '__main__':
  import gearup

  gearup.gearup(main).with_config('config/strawtubes.yaml')()