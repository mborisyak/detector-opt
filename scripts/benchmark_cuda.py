"""
Benchmark and correctness check: C solver vs CUDA solver.

Geometry is taken from config/detector/straw.yaml:
  - 4 stations × 4 views/station × 2 layers/view = 32 layers
  - 200 straws per layer, pitch 2 cm  → straw radius 1 cm
  - straw half-length 200 cm (height), half-width 200 cm (width)
  - view stereo angles [0, 0.0798, -0.0798, 0] rad
  - dt = 0.1 ns, B_sigma = 300 cm, z0 = 8957 cm
  - max_particles = 2, p_spawn = 0 (no secondaries)
"""

import ctypes, os, sys, time
import numpy as np

# ── load shared libraries ────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")


import detopt.detector.straw_detector as _c_mod

_cuda_so = os.path.join(ROOT, "detopt/detector/straw_detector_cuda.so")
import importlib.util, importlib.machinery

spec = importlib.util.spec_from_file_location("straw_detector_cuda", _cuda_so)
_cu_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_cu_mod)


# ── geometry builder (mirrors straw.yaml) ───────────────────────────────────
def make_geometry(n_batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    station_z = np.array([8407.0, 8607.0, 9307.0, 9507.0])
    view_angles = np.array([0.0, 0.0798, -0.0798, 0.0])
    layer_z_gap = 1.732  # cm between the two layers in a view
    view_z_gap = 5.0  # cm between consecutive views

    layer_zs, layer_angs = [], []
    for sz in station_z:
        for vi, va in enumerate(view_angles):
            z_view = sz + vi * view_z_gap
            for li in range(2):
                layer_zs.append(z_view + li * layer_z_gap)
                layer_angs.append(va)

    n_layers = len(layer_zs)  # 32
    n_straws = 200
    height = 200.0  # cm  (half-height → straw r = 1 cm)
    width = 200.0  # cm  (half-width)

    layers = np.tile(np.array(layer_zs, dtype=np.float32), (n_batch, 1))
    angles = np.tile(np.array(layer_angs, dtype=np.float32), (n_batch, 1))
    widths = np.full((n_batch, n_layers), width, dtype=np.float32)
    heights = np.full((n_batch, n_layers), height, dtype=np.float32)

    # Random B field between 0 and 0.15 T per event
    Bs = rng.uniform(0.05, 0.15, size=n_batch).astype(np.float32)
    z0s = np.full(n_batch, 8957.0, dtype=np.float32)
    B_sigmas = np.full(n_batch, 300.0, dtype=np.float32)

    return layers, angles, widths, heights, Bs, z0s, B_sigmas, n_layers, n_straws


def make_particles(n_batch: int, n_particles: int, seed: int = 0):
    """Synthetic HNL-like decay daughters starting upstream of the detector."""
    rng = np.random.default_rng(seed)

    # Start just before the first station (z ≈ 8200 cm)
    z_start = 8200.0
    positions = np.zeros((n_batch, n_particles, 3), dtype=np.float32)
    positions[:, :, 0] = rng.uniform(-5.0, 5.0, (n_batch, n_particles))
    positions[:, :, 1] = rng.uniform(-5.0, 5.0, (n_batch, n_particles))
    positions[:, :, 2] = z_start

    # Momenta: muon-like, mostly forward, ~200–400 MeV/c
    p_mag = rng.uniform(200.0, 400.0, (n_batch, n_particles)).astype(np.float32)
    phi = rng.uniform(0, 2 * np.pi, (n_batch, n_particles)).astype(np.float32)
    theta = rng.uniform(0.005, 0.03, (n_batch, n_particles)).astype(np.float32)
    momenta = np.zeros((n_batch, n_particles, 3), dtype=np.float32)
    momenta[:, :, 0] = p_mag * np.sin(theta) * np.cos(phi)
    momenta[:, :, 1] = p_mag * np.sin(theta) * np.sin(phi)
    momenta[:, :, 2] = p_mag * np.cos(theta)

    # Muon mass 105.66 MeV/c², charge ±1
    masses = np.full((n_batch, n_particles), 105.66, dtype=np.float32)
    charges = np.where(rng.integers(0, 2, (n_batch, n_particles)), 1.0, -1.0).astype(np.float32)
    times = np.zeros((n_batch, n_particles), dtype=np.float32)

    return positions, momenta, masses, charges, times


def alloc_sparse(n_batch, n_particles, n_layers, max_particles):
    max_hits = 2 * n_batch * max_particles * n_layers
    return dict(
        events=np.zeros(max_hits, dtype=np.int32),
        particles=np.zeros(max_hits, dtype=np.int32),
        layers=np.zeros(max_hits, dtype=np.int32),
        straws=np.zeros(max_hits, dtype=np.int32),
        values=np.zeros(max_hits, dtype=np.float32),
        r_mm=np.zeros(max_hits, dtype=np.float32),
        t0=np.zeros(max_hits, dtype=np.float32),
        hit_pos=np.zeros((max_hits, 3), dtype=np.float32),
        count=np.zeros(1, dtype=np.int32),
    )


def call_solver(
    mod,
    pos,
    mom,
    masses,
    charges,
    times,
    Bs,
    z0s,
    B_sigmas,
    layers,
    widths,
    heights,
    angles,
    n_steps,
    dt,
    n_batch,
    n_particles,
    n_layers,
    n_straws,
    max_particles,
    sparse,
):
    mod.solve(
        pos,
        mom,
        masses,
        charges,
        times,
        Bs,
        z0s,
        B_sigmas,
        n_steps,
        dt,
        int(n_batch),
        int(n_particles),
        int(n_layers),
        int(n_straws),
        layers,
        widths,
        heights,
        angles,
        None,  # no trajectories
        sparse["events"],
        sparse["particles"],
        sparse["layers"],
        sparse["straws"],
        sparse["values"],
        sparse["r_mm"],
        sparse["t0"],
        sparse["hit_pos"],
        sparse["count"],
        0.0,  # p_spawn_single
        0.0,  # p_spawn_pair
        0.01,  # E_sec_MeV
        int(max_particles),
    )
    return int(sparse["count"][0])


# ── correctness check ────────────────────────────────────────────────────────
def check_correctness(n_batch=8, n_particles=2, seed=42):
    print("=" * 60)
    print("CORRECTNESS CHECK")
    print("=" * 60)

    layers, angles, widths, heights, Bs, z0s, B_sigmas, n_layers, n_straws = make_geometry(n_batch, seed)
    pos, mom, masses, charges, times = make_particles(n_batch, n_particles, seed)

    n_steps = 700
    dt = 0.1
    max_particles = n_particles

    sp_c = alloc_sparse(n_batch, n_particles, n_layers, max_particles)
    sp_cu = alloc_sparse(n_batch, n_particles, n_layers, max_particles)

    n_c = call_solver(
        _c_mod,
        pos,
        mom,
        masses,
        charges,
        times,
        Bs,
        z0s,
        B_sigmas,
        layers,
        widths,
        heights,
        angles,
        n_steps,
        dt,
        n_batch,
        n_particles,
        n_layers,
        n_straws,
        max_particles,
        sp_c,
    )
    n_cu = call_solver(
        _cu_mod,
        pos,
        mom,
        masses,
        charges,
        times,
        Bs,
        z0s,
        B_sigmas,
        layers,
        widths,
        heights,
        angles,
        n_steps,
        dt,
        n_batch,
        n_particles,
        n_layers,
        n_straws,
        max_particles,
        sp_cu,
    )

    print(f"  C    hits: {n_c}")
    print(f"  CUDA hits: {n_cu}")

    # Sort both by (event, particle, layer, straw) for stable comparison
    def sort_hits(sp, n):
        idx = np.lexsort((sp["straws"][:n], sp["layers"][:n], sp["particles"][:n], sp["events"][:n]))
        return {k: v[:n][idx] for k, v in sp.items() if k != "count"}

    sc = sort_hits(sp_c, n_c)
    scu = sort_hits(sp_cu, n_cu)

    ok = True
    if n_c != n_cu:
        print(f"  MISMATCH: hit counts differ ({n_c} vs {n_cu})")
        ok = False
    else:
        for key in ("events", "particles", "layers", "straws"):
            if not np.array_equal(sc[key], scu[key]):
                print(f"  MISMATCH in '{key}'")
                ok = False

        if ok:
            r_diff = np.abs(sc["r_mm"] - scu["r_mm"])
            t0_diff = np.abs(sc["t0"] - scu["t0"])
            print(f"  r_mm  max|Δ| = {r_diff.max():.6f} mm")
            print(f"  t0    max|Δ| = {t0_diff.max():.6f} ns")

    if ok:
        print("  PASS – outputs match")
    else:
        # Print first few differing hits for debugging
        print("\n  First 10 C hits:")
        for i in range(min(10, n_c)):
            print(
                f"    ev={sp_c['events'][i]} p={sp_c['particles'][i]} "
                f"l={sp_c['layers'][i]} s={sp_c['straws'][i]} "
                f"r={sp_c['r_mm'][i]:.3f} t0={sp_c['t0'][i]:.3f}"
            )
        print("\n  First 10 CUDA hits:")
        for i in range(min(10, n_cu)):
            print(
                f"    ev={sp_cu['events'][i]} p={sp_cu['particles'][i]} "
                f"l={sp_cu['layers'][i]} s={sp_cu['straws'][i]} "
                f"r={sp_cu['r_mm'][i]:.3f} t0={sp_cu['t0'][i]:.3f}"
            )
    return ok


# ── benchmark ────────────────────────────────────────────────────────────────
def benchmark(
    n_batch_list=(1, 8, 64, 256, 1024),
    n_particles=2,
    n_steps=700,
    dt=0.1,
    n_warmup=2,
    n_repeat=5,
    seed=0,
):
    print()
    print("=" * 60)
    print("BENCHMARK")
    print(f"  n_particles={n_particles}, n_steps={n_steps}, dt={dt} ns")
    print(f"  warmup={n_warmup}, repeat={n_repeat}")
    print("=" * 60)
    print(f"{'n_batch':>8}  {'C (ms)':>10}  {'CUDA (ms)':>10}  {'speedup':>8}")
    print("-" * 45)

    for n_batch in n_batch_list:
        layers, angles, widths, heights, Bs, z0s, B_sigmas, n_layers, n_straws = make_geometry(n_batch, seed)
        pos, mom, masses, charges, times = make_particles(n_batch, n_particles, seed)
        max_particles = n_particles

        def run_c():
            sp = alloc_sparse(n_batch, n_particles, n_layers, max_particles)
            return call_solver(
                _c_mod,
                pos,
                mom,
                masses,
                charges,
                times,
                Bs,
                z0s,
                B_sigmas,
                layers,
                widths,
                heights,
                angles,
                n_steps,
                dt,
                n_batch,
                n_particles,
                n_layers,
                n_straws,
                max_particles,
                sp,
            )

        def run_cu():
            sp = alloc_sparse(n_batch, n_particles, n_layers, max_particles)
            return call_solver(
                _cu_mod,
                pos,
                mom,
                masses,
                charges,
                times,
                Bs,
                z0s,
                B_sigmas,
                layers,
                widths,
                heights,
                angles,
                n_steps,
                dt,
                n_batch,
                n_particles,
                n_layers,
                n_straws,
                max_particles,
                sp,
            )

        for _ in range(n_warmup):
            run_c()
            run_cu()

        t_c = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            run_c()
            t_c.append(time.perf_counter() - t0)

        t_cu = []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            run_cu()
            t_cu.append(time.perf_counter() - t0)

        mc = np.median(t_c) * 1e3
        mcu = np.median(t_cu) * 1e3
        print(f"{n_batch:>8}  {mc:>10.2f}  {mcu:>10.2f}  {mc/mcu:>8.2f}x")


if __name__ == "__main__":
    ok = check_correctness(n_batch=8, n_particles=2)
    benchmark(n_batch_list=[1, 8, 64, 256, 1024])
