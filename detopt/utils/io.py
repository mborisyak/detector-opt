import os

import flax
import jax.tree
import jax.numpy as jnp
from flax import nnx

import orbax.checkpoint as ocp

from . import config as config_utils

__all__ = [
    "get_checkpointer",
    "save_model",
    "restore_model",
    "restore_aux",
    "load_design",
    "check_bo_results",
    "check_scaled_design",
    "save_design",
    "restore_state",
    "save_state",
    "save_training_checkpoint",
    "restore_design",
    "restore_training_checkpoint",
    "save_checkpoint",
    "restore_config",
    "restore_checkpoint",
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
    """Read a design file written by :func:`save_design` and return it SCALED to ``[0, 1]``, the
    space every optimiser searches.

    The file must be the TAGGED form ``{"space": "nominal", "design": [...]}``. A bare JSON list is
    the pre-migration format and is refused rather than converted, because a bounds check cannot
    tell the two apart: those files hold ENCODED N(0,1) vectors (~0 = the centre of every range),
    and wherever a detector's bounds start at zero -- ``enzyme_fraction`` ``[0, 1]``, ``temperature``
    ``[0, 100]`` -- scaling ~0 lands at ~0, comfortably inside ``[0, 1]``. It would read as the lower
    CORNER of every range with nothing whatsoever out of place. Hence the tag."""
    import json
    import numpy as np

    with open(design_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("space") != "nominal":
        raise ValueError(
            f"{design_path}: not a tagged NOMINAL design file. A bare list is the pre-migration "
            f"format, holding an ENCODED N(0,1) vector that is not a design in the current "
            f"parameterisation and cannot be converted -- re-run to regenerate it, or write "
            f'{{"space": "nominal", "design": [...]}} in physical units.'
        )
    arr = np.asarray(data["design"], dtype=np.float32)
    scaled = np.asarray(detector.to_scaled(arr), dtype=np.float32)
    if np.any(scaled < -0.01) or np.any(scaled > 1.01):
        raise ValueError(
            f"{design_path}: the design is outside this detector's bounds -- scaling it gives "
            f"[{scaled.min():.3g}, {scaled.max():.3g}], not [0, 1]."
        )
    return scaled


def check_scaled_design(design_scaled, where):
    """Reject a restored design that is not a SCALED vector in ``[0, 1]``.

    A checkpoint written before the encoded->scaled migration stores an ENCODED theta (N(0,1), so
    ~0 is the CENTRE of every range) together with design-optimiser moments accumulated at encoded
    gradient magnitudes. Restored onto the scaled cube the theta reads as the lower CORNER and the
    moments mis-scale the first steps -- neither raises on its own. A finite tolerance is allowed
    because an un-clipped design step can leave the cube slightly before being used."""
    import numpy as np

    theta = np.asarray(design_scaled, dtype=np.float64)
    if np.any(theta < -0.5) or np.any(theta > 1.5):
        raise ValueError(
            f"{where}: restored design is not SCALED -- it lies in [{theta.min():.3g}, {theta.max():.3g}], "
            f"not [0, 1]. A pre-migration checkpoint (encoded theta + optimiser moments at encoded "
            f"magnitudes) looks exactly like this and cannot be resumed; start a fresh run."
        )
    return design_scaled


def check_bo_results(results, path):
    """Reject a ``results.json`` written before the encoded->scaled migration.

    Pre-migration runs stored the searched design as ``"x_encoded"`` in N(0,1) units. Those numbers
    are not scaled designs: pushing them through ``to_nominal`` extrapolates linearly outside the
    design box, so the file must not simply be re-keyed. The ``"design"`` column is physical and is
    the only safe re-entry point. Returns ``results`` unchanged when the file is current."""
    if len(results) > 0 and "x_scaled" not in results[0]:
        found = "x_encoded" if "x_encoded" in results[0] else "neither x_scaled nor x_encoded"
        raise ValueError(
            f"{path}: pre-migration BO results ({found}). The searched space is now the SCALED cube "
            f"[0, 1]^d, and the stored N(0,1) vectors do not convert -- re-run, or re-enter through "
            f"the physical 'design' column."
        )
    return results


def save_design(detector, design_path, design_scaled):
    """Write the SCALED design ``design_scaled`` out in NOMINAL (physical) units, TAGGED with the
    space it is in. Physical units are the only parameterisation-independent form, so a later
    re-parameterisation cannot invalidate the file; the tag is what lets :func:`load_design` refuse
    a pre-migration file instead of silently misreading it."""
    import json
    import numpy as np

    os.makedirs(os.path.dirname(design_path), exist_ok=True)

    nominal = np.asarray(detector.flatten_design(detector.to_nominal(design_scaled)), dtype=np.float32)
    with open(design_path, "w") as f:
        json.dump({"space": "nominal", "design": nominal.tolist()}, f, indent=2)


def save_training_checkpoint(manager, step, *, config, parameters, state, design, aux=None):
    """Save a design's converged network + design into a (per-design) ``manager`` at ``step``.

    ``manager`` is a per-design :func:`get_checkpointer` (one per design iteration), written
    ONCE at convergence with ``step`` = the design's epoch count -- so the checkpoint holds
    exactly the state whose loss the run reported, which is what every reader asks for via
    ``latest_step()``. (``max_to_keep`` therefore never prunes: there is only one step.)
    The trained network (``parameters`` / ``state``, flattened to pure dicts) and
    the ``design`` are saved as *separate* orbax items, so :func:`restore_design`
    reads the small design tree alone -- quick access without the network. The model
    ``config`` is stored in the SAME ``"config"`` field as :func:`save_checkpoint` (read
    back via :func:`restore_config`) so a test/eval run rebuilds the exact architecture.
    (No optimizer state: each design trains a fresh network -- there is no resume here.)
    """
    manager.save(
        step,
        args=ocp.args.Composite(
            config=ocp.args.JsonSave(config),
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
    ``design`` is the ``{"scaled", "physical"}`` tree; ``aux`` the saved metrics.
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


# --------------------------------------------------------------------------- #
# Uniform checkpoint machinery (used by ALL active training/eval scripts).
#
# A checkpoint bundles, as separate orbax items: the model config(s) (JSON, so
# resume/test rebuilds the EXACT architecture the checkpoint was trained with --
# config files drift, the checkpoint does not), each trained network
# (parameters/state/optimizer_state), the optimized design, and an aux pytree.
# `save_checkpoint` writes whichever are provided; `restore_config` reads the
# config alone (cheap, needed BEFORE a model exists); `restore_checkpoint`
# loads the weights into freshly built models. New training -> config from the
# config file; resume / test -> config from the checkpoint.
# --------------------------------------------------------------------------- #
def _model_blob(params, state, optimizer_state):
    """Live nnx ``(params, state, opt_state)`` -> the pure-dict blob orbax stores."""
    return {
        "parameters": nnx.to_pure_dict(params),
        "state": nnx.to_pure_dict(state),
        "optimizer_state": jax.tree.leaves(optimizer_state),
    }


def _load_model_blob(blob, params0, state0, optimizer):
    """Stored blob + a freshly built model's abstract ``(params0, state0)`` and its ``optimizer``
    -> live ``(params, state, opt_state)`` (pure dicts loaded back into nnx state)."""
    params = nnx.eval_shape(lambda: params0)
    nnx.replace_by_pure_dict(params, blob["parameters"])
    state = nnx.eval_shape(lambda: state0)
    nnx.replace_by_pure_dict(state, blob["state"])
    opt_state = jax.tree.unflatten(jax.tree.structure(optimizer.init(params)), blob["optimizer_state"])
    return params, state, opt_state


def save_checkpoint(manager, step, *, config, design=None, aux=None, **models):
    """Uniform checkpoint save.

    ``config`` -- a JSON-serialisable dict of the MODEL configs (e.g. ``{"regressor": ...}`` or
    ``{"regressor": ..., "discriminator": ...}``), stored as a SEPARATE field so resume/test rebuilds
    the exact architecture. ``**models`` -- one entry per network, ``name=(params, state, opt_state)``
    (e.g. ``regressor=(...)``, ``discriminator=(...)``); each is written as its own item. ``design`` --
    ``(theta, design_opt_state)`` for the design-optimisation loops. ``aux`` -- any extra pytree
    (history / metadata). Only the provided items are written.
    """
    items = {"config": ocp.args.JsonSave(config)}
    for name, triple in models.items():
        if triple is not None:
            items[name] = ocp.args.PyTreeSave(_model_blob(*triple))
    if design is not None:
        theta, design_opt_state = design
        items["design"] = ocp.args.PyTreeSave(
            {"theta": theta, "optimizer_state": None if design_opt_state is None else jax.tree.leaves(design_opt_state)})
    if aux is not None:
        items["aux"] = ocp.args.PyTreeSave(aux)
    manager.save(step, args=ocp.args.Composite(**items))


def restore_config(manager, step=None):
    """The MODEL config dict saved alongside the checkpoint, or ``None`` if there is no checkpoint or it
    predates config-saving. Read on resume/test to rebuild the architecture BEFORE loading the weights."""
    if step is None:
        step = manager.latest_step()
    if step is None:
        return None
    try:
        return manager.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]
    except KeyError:
        return None  # checkpoint predates config-saving


def restore_checkpoint(manager, step=None, *, design=None, aux=False, **models):
    """Uniform restore matching :func:`save_checkpoint`. ``**models`` -- one entry per network,
    ``name=(params0, state0, optimizer)`` (a freshly built model's abstract state + its optimizer) ->
    restored live ``(params, state, opt_state)``. ``design`` -- the design optax optimizer -> restored
    ``(theta, design_opt_state)``. ``aux`` True -> include the saved aux pytree. Returns a dict keyed by
    the requested item names (``"regressor"``, ``"design"``, ...)."""
    if step is None:
        step = manager.latest_step()
    spec = {name: ocp.args.PyTreeRestore() for name, triple in models.items() if triple is not None}
    if design is not None:
        spec["design"] = ocp.args.PyTreeRestore()
    if aux:
        spec["aux"] = ocp.args.PyTreeRestore()
    data = manager.restore(step, args=ocp.args.Composite(**spec))
    out = {name: _load_model_blob(data[name], *triple) for name, triple in models.items() if triple is not None}
    if design is not None:
        theta = jnp.asarray(data["design"]["theta"], jnp.float32)
        dos = data["design"]["optimizer_state"]
        out["design"] = (theta, None if dos is None else
                         jax.tree.unflatten(jax.tree.structure(design.init(theta)), dos))
    if aux:
        out["aux"] = data["aux"]
    return out


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
