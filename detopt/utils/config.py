from typing import Any, TypeVar

__all__ = ["split", "extract", "optimizer", "resolve_device", "load_config"]


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
        (name,) = config.keys()
        arguments = config[name]
        return name, arguments
    else:
        raise ValueError(
            f"config entry must contain a dictionary with exactly one field (name of the object), "
            f'got {", ".join(config.keys())}'
        )


T = TypeVar("T")


def extract(config: dict[str, dict[str, Any]], library: dict[str, T]) -> tuple[T, dict[str, Any]]:
    name, arguments = split(config)
    if name not in library:
        raise ValueError(f"{name} does not appear to be a valid object")

    return library[name], arguments


def optimizer(config: dict[str, Any], n_total_steps: int | None=None):
    import optax

    name, arguments = split(config)
    if 'learning_rate' in arguments:
        if isinstance(arguments['learning_rate'], dict):
            lr_name, lr_arguments = split(arguments['learning_rate'])
            arguments['learning_rate'] = getattr(optax, lr_name)(**lr_arguments, decay_steps=n_total_steps)

    return getattr(optax, name)(**arguments)


def resolve_device(name):
    """Resolve a JAX device by platform name (e.g. ``"cuda"``); ``None`` -> default."""
    if name is None:
        return None
    import jax

    devices = jax.devices(str(name).lower())
    if not devices:
        raise RuntimeError(f"No JAX devices for backend {name!r}")
    return devices[0]


def load_config(path):
    """Load a YAML (``.yaml``/``.yml``) or JSON config file into a dict."""
    import json

    path = str(path)
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(f)
        return json.load(f)
