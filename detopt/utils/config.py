from typing import Any, TypeVar

__all__ = ["split", "extract", "optimizer", "resolve_device", "load_config", "override"]


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


def optimizer(config: dict[str, Any], n_total_steps: int | None = None):
    import optax

    name, arguments = split(config)
    if "learning_rate" in arguments:
        if isinstance(arguments["learning_rate"], dict):
            lr_name, lr_arguments = split(arguments["learning_rate"])
            arguments["learning_rate"] = getattr(optax, lr_name)(**lr_arguments, decay_steps=n_total_steps)

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


def override(config, assignments):
    """Apply ``dotted.key=value`` assignments to a loaded config, in place, and return it.

    The value is parsed as YAML, so scalars, lists and mappings all work
    (``enzyme.parameters.T_melting='[25.0, 75.0]'``). An assignment to a key the config does not
    already define is an ERROR rather than a new entry: these come from a command line, where a
    typo would otherwise be silently ignored and the sweep would report the unchanged config.
    """
    import yaml

    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if separator != "=":
            raise ValueError(f"expected dotted.key=value, got {assignment!r}")
        node = config
        *path, leaf = key.split(".")
        for step in path:
            node = node[step]
        if leaf not in node:
            raise KeyError(f"{key!r} is not a config key ({leaf!r} not in {sorted(node)})")
        node[leaf] = yaml.safe_load(value)
    return config
