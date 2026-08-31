#!/usr/bin/env python3
"""Plot the enzymatic system the ``enzyme`` detector simulates.

Draws ``n_draws`` enzymes from the detector's prior and, for each, plots the reaction (``[A]`` and
``[C]`` against time at that enzyme's own optimal temperature; ``[B]`` and ``[D]`` follow by
stoichiometry) together with the objective the target maximises (conversion after ``duration``
against temperature, with the optimal temperature marked).

Both the figure and the numbers behind it are written out, so the plot regenerates from JSON alone::

    python scripts/plot_enzyme.py =enzyme output=output/enzyme
    python scripts/plot_enzyme.py =enzyme output=output/enzyme replot=true   # from system.json only
"""

import json
import os

import matplotlib

matplotlib.use("AGG")

import numpy as np

import jax
import jax.numpy as jnp

import detopt
from detopt.utils.viz.enzyme import plot_system


# Temperatures plotted for the conversion curve. A PLOTTING resolution: the detector itself no longer
# tabulates temperatures (it locates the peak by golden-section search), so the curve behind the peak
# is sampled here, densely enough to draw smoothly.
N_TEMPERATURES = 129


def _sample(detector, index, n_times, temperatures):
    """One drawn enzyme: its calibrated turnover, optimal temperature, the reference mixture's
    ``(A, C)`` trajectory at that temperature, and the conversion-vs-temperature curve.

    The trajectory reuses the detector's own Euler integrator, asking it for ``n_times`` intervals so
    the returned extents ARE the sampled trajectory -- the same solver the events go through, not a
    second implementation.
    """
    key_parameters, key_half_time, _ = jax.random.split(jax.random.PRNGKey(index), 3)
    parameters = detector._draw_parameters(key_parameters)
    low, high = detector.half_time_bounds
    half_time = jnp.exp(jax.random.uniform(key_half_time, (), minval=np.log(low), maxval=np.log(high)))
    log_k0_cat = detector._calibrate(parameters, half_time)
    optimal, _ = detector._optimal_temperature(parameters, log_k0_cat)

    A0, B0, E0 = detector._initial_state(detector.objective_fraction)
    extent, _ = detector._integrate(
        A0, B0, E0, optimal, parameters, log_k0_cat,
        n_steps=max(1, detector.n_objective_steps // n_times), n_intervals=n_times,
    )
    # The conversion curve behind the peak (one integration per plotted temperature).
    conversion = jax.vmap(
        lambda T: detector._integrate(A0, B0, E0, T, parameters, log_k0_cat,
                                  n_steps=detector.n_objective_steps, n_intervals=1)[0][-1]
    )(temperatures) / jnp.minimum(A0, B0)

    return {
        "A": A0 - extent,
        "C": extent,
        "conversion": conversion,
        "optimal_temperature": optimal,
        "T_melting": parameters["T_melting"],
        "half_time": half_time,
        "log_k0_cat": log_k0_cat,
        "initial_A": jnp.asarray(A0, jnp.float32),
    }


def plot(output, n_draws: int = 10, n_times: int = 64, replot: bool = False, **config):
    os.makedirs(output, exist_ok=True)
    json_path = os.path.join(output, "system.json")

    if replot:  # regenerate the figure from the stored numbers, no simulation
        with open(json_path) as f:
            system = json.load(f)
        print(f"Replotting {len(system['draws'])} draws from {json_path}")
    else:
        detector = detopt.detector.from_config(config["detector"])
        n_draws, n_times = int(n_draws), int(n_times)
        temperatures = jnp.linspace(*detector.temperature_bounds, N_TEMPERATURES, dtype=jnp.float32)
        print(f"Drawing {n_draws} enzymes from the prior ({N_TEMPERATURES} plotted temperatures)...")
        drawn = jax.jit(jax.vmap(lambda i: _sample(detector, i, n_times, temperatures)))(
            jnp.arange(n_draws, dtype=jnp.int32)
        )

        times = np.asarray(detector.duration / n_times * np.arange(1, n_times + 1), dtype=float)
        system = {
            "times": times.tolist(),
            "temperatures": np.asarray(temperatures, dtype=float).tolist(),
            "duration": detector.duration,
            "objective_fraction": detector.objective_fraction,
            "loss_label": detector.loss_label(),
            "draws": [
                {
                    "A": np.asarray(drawn["A"][i], dtype=float).tolist(),
                    "C": np.asarray(drawn["C"][i], dtype=float).tolist(),
                    "conversion": np.asarray(drawn["conversion"][i], dtype=float).tolist(),
                    "optimal_temperature": float(drawn["optimal_temperature"][i]),
                    "T_melting": float(drawn["T_melting"][i]),
                    "half_time": float(drawn["half_time"][i]),
                    "log_k0_cat": float(drawn["log_k0_cat"][i]),
                    "initial_A": float(drawn["initial_A"][i]),
                }
                for i in range(n_draws)
            ],
        }
        with open(json_path, "w") as f:
            json.dump(system, f, indent=2)
        print(f"  -> {json_path}")

        optimal = np.array([d["optimal_temperature"] for d in system["draws"]])
        melting = np.array([d["T_melting"] for d in system["draws"]])
        print(f"  T*: {np.round(optimal, 1).tolist()}")
        print(f"  T* - T_melting: mean {np.mean(optimal - melting):+.2f} C")

    plot_system(system, os.path.join(output, "system.png"))
    return system


if __name__ == "__main__":
    import gearup

    gearup.gearup(plot).with_config("config/root.yaml")()
