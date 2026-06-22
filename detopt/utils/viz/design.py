"""Universal, detector-invariant design-trajectory plot for the optimization scripts.

The physical design is a named dict; each detector exposes ``design_spec`` (``{name: shape}``) and
``design_bounds() -> {name: (lo, hi)}``. :func:`design_trajectory_meta`
carries that (json-able) in the script ``aux``/``trajectory.json``, and :func:`plot_design_trajectory`
draws **one subplot per design key** (each key's components vs epoch, in physical units, y-limited to
the key's bound) into a given figure/subfigure -- detector-invariant, no per-detector plot code.
"""

import numpy as np
from matplotlib import colormaps

__all__ = ["design_trajectory_meta", "plot_design_trajectory"]


def design_trajectory_meta(detector):
    """The detector's ``design_spec`` + ``design_bounds`` as json-able metadata, aligned with the
    flat physical-design vector the trajectory stores. ``design_spec`` is a ``Design`` namedtuple of
    ``jax.ShapeDtypeStruct`` (field order = flat-vector order); ``design_bounds`` is keyed by field name."""
    spec_record = detector.design_spec()
    spec = [[name, int(np.prod(leaf.shape))] for name, leaf in zip(spec_record._fields, spec_record)]
    bounds = {name: [float(lo), float(hi)] for name, (lo, hi) in detector.design_bounds().items()}
    return {"spec": spec, "bounds": bounds}


def plot_design_trajectory(fig, designs, meta=None):
    """Draw ONE subplot per design key into ``fig`` (a Figure or SubFigure): each key's components
    plotted vs epoch in physical units, y-limited to the key's design bound."""
    designs = np.asarray(designs)
    if designs.ndim != 2 or designs.shape[0] == 0:
        return
    meta = meta or {}
    spec = meta.get("spec") or [["design", designs.shape[1]]]  # fallback: one group
    bounds = meta.get("bounds") or {}
    ep = np.arange(designs.shape[0])
    cmap = colormaps["tab10"]

    axes = np.atleast_1d(fig.subplots(1, len(spec), squeeze=False)[0])
    col = 0
    for ax, (name, size) in zip(axes, spec):
        block = designs[:, col : col + size]
        col += size
        for j in range(size):
            ax.plot(ep, block[:, j], ".-", color=cmap(j % 10), label=(name if size == 1 else f"{name}[{j}]"))
        ax.set(title=name, xlabel="epoch", ylabel=name)
        lo, hi = bounds.get(name, (None, None))
        if lo is not None and hi is not None and hi > lo:
            ax.set_ylim(lo, hi)
        if size > 1:
            ax.legend(fontsize=7, ncol=max(1, (size + 3) // 4), loc="best")
