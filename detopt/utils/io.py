import os

import flax
import jax.tree
from flax import nnx

import orbax.checkpoint as ocp

from . import config as config_utils

__all__ = [
    "get_checkpointer",
    "save_model",
    "restore_model",
    "restore_aux",
    "load_design",
    "save_design",
    "restore_state",
    "save_state",
    "save_training_checkpoint",
    "restore_design",
    "restore_training_checkpoint",
]


def get_checkpointer(path):
    import absl.logging

    absl.logging.set_verbosity(absl.logging.ERROR)

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=1,
    )
    manager = ocp.CheckpointManager(path, options=options)
    return manager


def restore_aux(manager: ocp.CheckpointManager):
    data = manager.restore(manager.latest_step(), args=ocp.args.Composite(aux=ocp.args.PyTreeRestore()))

    return data["aux"]


def load_design(detector, design_path):
    import json
    import numpy as np

    with open(design_path, "r") as f:
        data = json.load(f)
    # Accept either a {positions, angles, magnetic_strength} dict or a flat
    # physical design array, then encode into unconstrained space.
    if isinstance(data, dict):
        data = detector.layer_design_to_array(data)
    arr = np.asarray(data, dtype=np.float32)
    return np.asarray(detector.encode_design(arr), dtype=np.float32)


def save_design(detector, design_path, design):
    import json
    import numpy as np

    os.makedirs(os.path.dirname(design_path), exist_ok=True)

    decoded = np.asarray(detector.flatten_design(detector.decode_design(design)), dtype=np.float32)
    with open(design_path, "w") as f:
        json.dump(decoded.tolist(), f, indent=2)


def save_training_checkpoint(manager, step, *, parameters, state, design, aux=None):
    """Save one epoch's network + design into a (per-design) ``manager`` at ``step``.

    ``manager`` is a per-design :func:`get_checkpointer` (one per design iteration),
    so saving every epoch prunes to the manager's ``max_to_keep`` most recent epochs.
    The trained network (``parameters`` / ``state``, flattened to pure dicts) and
    the ``design`` are saved as *separate* orbax items, so :func:`restore_design`
    reads the small design tree alone -- quick access without the network.
    """
    manager.save(
        step,
        args=ocp.args.Composite(
            regressor=ocp.args.PyTreeSave(
                {
                    "parameters": nnx.to_pure_dict(parameters),
                    "state": nnx.to_pure_dict(state),
                }
            ),
            design=ocp.args.PyTreeSave(design),
            aux=ocp.args.PyTreeSave(aux if aux is not None else {}),
        ),
    )


def restore_design(manager, step=None):
    """Restore only the design tree at ``step`` (latest if ``None``) -- no network load."""
    if step is None:
        step = manager.latest_step()
    return manager.restore(step, args=ocp.args.Composite(design=ocp.args.PyTreeRestore()))["design"]


def restore_training_checkpoint(manager, step=None):
    """Restore ``(parameters, state, design, aux)`` saved by :func:`save_training_checkpoint`.

    ``parameters`` / ``state`` come back as pure dicts -- load them into an abstract
    nnx state (``nnx.split`` of a freshly built module) with ``nnx.replace_by_pure_dict``.
    ``design`` is the ``{"encoded", "physical"}`` tree; ``aux`` the saved metrics.
    """
    if step is None:
        step = manager.latest_step()
    data = manager.restore(
        step,
        args=ocp.args.Composite(
            regressor=ocp.args.PyTreeRestore(),
            design=ocp.args.PyTreeRestore(),
            aux=ocp.args.PyTreeRestore(),
        ),
    )
    reg = data["regressor"]
    return reg["parameters"], reg["state"], data["design"], data["aux"]


def save_model(parameters, state, optimizer_state):
    if parameters is None:
        return dict(parameters=None, state=None, optimizer_state=None)

    optimizer_state = jax.tree.leaves(optimizer_state)
    return dict(parameters=parameters, state=state, optimizer_state=optimizer_state)


def save_state(
    step,
    manager: ocp.CheckpointManager,
    design,
    design_optimizer_state=None,
    regressor_parameters=None,
    regressor_state=None,
    regressor_optimizer_state=None,
    generator_parameters=None,
    generator_state=None,
    generator_optimizer_state=None,
    discriminator_parameters=None,
    discriminator_state=None,
    discriminator_optimizer_state=None,
    *,
    aux,
):
    if design_optimizer_state is not None:
        design_optimizer_state = jax.tree.leaves(design_optimizer_state)

    manager.save(
        step,
        args=ocp.args.Composite(
            ### because saving a standalone array is difficult
            design=ocp.args.PyTreeSave({"design": design, "optimizer_state": design_optimizer_state}),
            regressor=ocp.args.PyTreeSave(save_model(regressor_parameters, regressor_state, regressor_optimizer_state)),
            generator=ocp.args.PyTreeSave(save_model(generator_parameters, generator_state, generator_optimizer_state)),
            discriminator=ocp.args.PyTreeSave(
                save_model(
                    discriminator_parameters,
                    discriminator_state,
                    discriminator_optimizer_state,
                )
            ),
            aux=ocp.args.PyTreeSave(aux),
        ),
    )


def restore_model(config, detector, restored, rngs):
    from .. import nn

    if config is None:
        return dict(
            model=None,
            optimizer=None,
            parameters=None,
            state=None,
            optimizer_state=None,
        )

    model = nn.from_config(detector, config=config["model"], rngs=rngs)
    model_def, initial_model_parameters, initial_model_state = nnx.split(model, nnx.Param, nnx.Variable)
    optimizer = config_utils.optimizer(config["optimizer"])

    if restored is None:
        parameters = nnx.to_pure_dict(initial_model_parameters)
        model_state = nnx.to_pure_dict(initial_model_state)
        optimizer_state = optimizer.init(parameters)
    else:
        parameters = restored["parameters"]
        model_state = restored["state"]
        optimizer_state = jax.tree.unflatten(jax.tree.structure(optimizer.init(parameters)), restored["optimizer_state"])

    return dict(
        model=model_def,
        optimizer=optimizer,
        parameters=parameters,
        state=model_state,
        optimizer_state=optimizer_state,
    )


def restore_state(manager, detector, config, *, rngs: nnx.Rngs, restore=True):
    if "optimizer" in config:
        design_optimizer = config_utils.optimizer(config["optimizer"])
    else:
        design_optimizer = None

    last_epoch = manager.latest_step()
    if last_epoch is not None and restore:
        starting_epoch = last_epoch + 1

        data = manager.restore(
            manager.latest_step(),
            args=ocp.args.Composite(
                design=ocp.args.PyTreeRestore(),
                regressor=ocp.args.PyTreeRestore(),
                generator=ocp.args.PyTreeRestore(),
                discriminator=ocp.args.PyTreeRestore(),
                aux=ocp.args.PyTreeRestore(),
            ),
        )
        ### design -> design because it is a standalone array
        design, design_optimizer_state = (
            data["design"]["design"],
            data["design"]["optimizer_state"],
        )

        if design_optimizer_state is not None:
            design_optimizer_state = jax.tree.unflatten(
                jax.tree.structure(design_optimizer.init(design)),
                design_optimizer_state,
            )

    else:
        starting_epoch = 0
        data = {}

        ### design -> design because it is a standalone array
        design = load_design(detector, config["initial_design"])
        if design_optimizer is None:
            design_optimizer_state = None
        else:
            design_optimizer_state = design_optimizer.init(design)

    regressor = restore_model(config.get("regressor", None), detector, data.get("regressor", None), rngs=rngs)

    generator = restore_model(config.get("generator", None), detector, data.get("generator", None), rngs=rngs)

    discriminator = restore_model(
        config.get("discriminator", None),
        detector,
        data.get("discriminator", None),
        rngs=rngs,
    )

    return dict(
        starting_epoch=starting_epoch,
        design=dict(
            design=design,
            optimizer=design_optimizer,
            optimizer_state=design_optimizer_state,
        ),
        regressor=regressor,
        generator=generator,
        discriminator=discriminator,
        aux=data.get("aux", None),
    )
