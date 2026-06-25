"""Deterministic, vectorized pseudo-randomness keyed per row: the same ``key`` reproduces the same draws
(common random numbers across designs / batch positions). Used by the no-data analytic event source
(:mod:`detopt.detector.stereo_straw`) and the simplified engine's drift smear
(:mod:`detopt.detector.propagation`), so an event index reproduces its event byte-for-byte.
"""

import numpy as np

__all__ = ["det_uniforms", "box_muller", "det_normals"]


def det_uniforms(key, k):
    """``(n, k)`` deterministic uniforms in ``[0, 1)`` from a per-row ``uint`` ``key`` -- a splitmix64
    finalizer keyed by the column index, so each ``(row, column)`` draw depends ONLY on the row key (a
    repeated key reproduces identical draws)."""
    key = np.asarray(key, np.uint64)[:, None]
    j = np.arange(k, dtype=np.uint64)[None, :]
    x = key * np.uint64(0x9E3779B97F4A7C15) + (j + np.uint64(1)) * np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(30); x = x * np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27); x = x * np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)  # / 2^53 -> [0, 1)


def box_muller(u):
    """``(n, 2p)`` uniforms -> ``(n, 2p)`` standard normals (Box-Muller over adjacent column pairs)."""
    u = np.asarray(u, np.float64)
    r = np.sqrt(-2.0 * np.log(np.maximum(u[:, 0::2], 1e-12)))
    theta = 2.0 * np.pi * u[:, 1::2]
    z = np.empty_like(u)
    z[:, 0::2] = r * np.cos(theta)
    z[:, 1::2] = r * np.sin(theta)
    return z


def det_normals(key, k):
    """``(n, k)`` deterministic standard normals from a per-row ``key`` (Box-Muller over an even number of
    :func:`det_uniforms`)."""
    return box_muller(det_uniforms(key, 2 * ((k + 1) // 2)))[:, :k]
