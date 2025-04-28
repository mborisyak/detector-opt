from typing import Any, TypeVar

__all__ = [
  'split',
  'extract',
  'optimizer'
]

def split(config: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
  """
    Python objects (models, optimizers etc) are configured in the following way in YAML:
        <name of the model / method / optimizer>:
          <arguments>

    This function checks that config follows this format and returns the name and the arguments.

    :param config: configuration for an object;
    :param library: a dictionary of named objects;
    :return: name of the object, arguments.
    """
  if len(config) == 1:
    name, = config.keys()
    arguments = config[name]
    return name, arguments
  else:
    raise ValueError(
      f'config entry must contain a dictionary with exactly one field (name of the object), '
      f'got {", ".join(config.keys())}'
    )

T = TypeVar('T')
def extract(config: dict[str, dict[str, Any]], library: dict[str, T]) -> tuple[T, dict[str, Any]]:
  name, arguments = split(config)
  if name not in library:
    raise ValueError(f'{name} does not appear to be a valid object')

  return library[name], arguments


def optimizer(config: dict[str, Any]):
  import optax

  name, arguments = split(config)

  return getattr(optax, name)(**arguments)
