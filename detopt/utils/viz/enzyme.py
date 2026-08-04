"""Plots of the enzymatic system behind the ``enzyme`` detector.

One figure, two blocks of small multiples over the SAME ten drawn enzymes (small multiples rather
than ten overlaid colours -- ten series in one axes is unreadable and needs a generated palette):

* top -- the reaction itself: ``[A]`` and ``[C]`` against time for the reference mixture at that
  enzyme's own optimal temperature. ``[B]`` and ``[D]`` follow by stoichiometry (``B = A`` when the
  stocks are equal, ``D = C``), so plotting them would just double the ink.
* bottom -- what the target is: conversion after ``duration`` against temperature, with the optimal
  temperature (the regression label) marked. Turnover rises with temperature while the enzyme
  unfolds, so the peak sits just below the denaturation edge.

Every plotted number is returned as plain lists so the caller can write them to JSON and regenerate
the figure without re-running the simulation.
"""

import numpy as np

__all__ = ["plot_system"]

# Categorical slots 1 and 2 of the validated default palette (blue / orange), plus slot 7 (violet)
# for the single-series conversion panels.
_A_COLOR = "#2a78d6"
_C_COLOR = "#eb6834"
_OBJECTIVE_COLOR = "#4a3aa7"
_INK = "#52514e"


def plot_system(system, out_path, *, title="Enzymatic system: A + B -> C + D via E"):
    """Render :func:`detopt.detector.enzyme.EnzymeDetector` sample trajectories + objective curves.

    ``system`` is the dict produced by ``scripts/plot_enzyme.py`` (also what it stores as JSON):
    ``times``, ``temperatures``, and a ``draws`` list of per-enzyme
    ``{A, C, conversion, optimal_temperature, T_melting, half_time, initial_A}``.
    """
    from matplotlib.figure import Figure

    draws = system["draws"]
    times = np.asarray(system["times"], dtype=np.float64)
    temperatures = np.asarray(system["temperatures"], dtype=np.float64)
    n = len(draws)
    columns = min(5, n)
    rows = int(np.ceil(n / columns))

    fig = Figure(figsize=(3.0 * columns, 2.5 * 2 * rows))
    axes = fig.subplots(2 * rows, columns, squeeze=False)

    for index, draw in enumerate(draws):
        row, column = divmod(index, columns)
        trajectory = axes[2 * row][column]
        objective = axes[2 * row + 1][column]

        # --- the reaction at this enzyme's optimal temperature
        A = np.asarray(draw["A"], dtype=np.float64)
        C = np.asarray(draw["C"], dtype=np.float64)
        trajectory.plot(times, A, lw=2, color=_A_COLOR, label="[A] substrate")
        trajectory.plot(times, C, lw=2, color=_C_COLOR, label="[C] product")
        trajectory.set_ylim(0.0, 1.05 * float(draw["initial_A"]))
        trajectory.set_xlim(0.0, times[-1])
        trajectory.set_title(
            f"#{index + 1}  T* = {draw['optimal_temperature']:.1f} C\n"
            f"$T_m$ = {draw['T_melting']:.1f} C, $t_{{1/2}}$ = {draw['half_time']:.2f} h",
            fontsize=9,
        )
        if column == 0:
            trajectory.set_ylabel("concentration (mM)", fontsize=9)
        if index == 0:  # >= 2 series -> a legend is always present; once is enough for the block
            trajectory.legend(loc="center right", fontsize=8, frameon=False)

        # --- the objective this enzyme's target maximises
        conversion = np.asarray(draw["conversion"], dtype=np.float64)
        objective.plot(temperatures, conversion, lw=2, color=_OBJECTIVE_COLOR)
        peak = float(draw["optimal_temperature"])
        objective.axvline(peak, color=_INK, ls="--", lw=1)
        objective.plot([peak], [conversion.max()], marker="o", ms=5, color=_OBJECTIVE_COLOR)
        objective.annotate(
            f"{peak:.1f} C", (peak, conversion.max()), textcoords="offset points", xytext=(-4, 4),
            ha="right", fontsize=8, color=_INK,
        )
        objective.set_ylim(0.0, 1.0)
        objective.set_xlim(temperatures[0], temperatures[-1])
        objective.set_xlabel("temperature (C)", fontsize=9)
        if column == 0:
            objective.set_ylabel(f"conversion at {times[-1]:.0f} h", fontsize=9)

        for ax in (trajectory, objective):
            ax.grid(True, alpha=0.25, lw=0.6)
            ax.tick_params(labelsize=8, colors=_INK)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

    for index in range(n, rows * columns):  # blank any unused column
        row, column = divmod(index, columns)
        axes[2 * row][column].axis("off")
        axes[2 * row + 1][column].axis("off")

    for row in range(rows):
        axes[2 * row][0].set_xlabel("time (h)", fontsize=9)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=120)
    print(f"  [plot] enzyme system -> {out_path}")
    return out_path
