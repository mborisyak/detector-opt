#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define SPACE_DIM 3

#define SLOW_VZ 1.0e-3
#define SLOW 1.0e-6

/* ROOT/VMC TMCProcess codes (== FairShip ShipMCTrack.GetProcID() -> mc_info[:,3]),
 * used to tag the particle that caused each hit and the debug MC tree. We only
 * emit the processes this solver actually models; the full enum is documented at
 * https://root.cern/doc/v606/TMCProcess_8h_source.html */
#define PROC_PRIMARY 0  /* kPPrimary  -- input HNL daughters */
#define PROC_DECAY 4    /* kPDecay    -- pi/K -> mu decay-in-flight */
#define PROC_PAIR 5     /* kPPair     -- gamma -> e+e- (conversion / brems-pair) */
#define PROC_DELTARAY 9 /* kPDeltaRay -- knock-on electron from ionisation */
#define PROC_NULL 31    /* kPNull     -- injected noise hit (no causing particle) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline npy_float square(npy_float x) { return x * x; }

inline npy_float f32_abs(npy_float x) { return x > 0.0 ? x : -x; }

inline npy_float f32_min(npy_float x, npy_float y) { return x > y ? y : x; }

inline npy_float f32_max(npy_float x, npy_float y) { return x > y ? x : y; }

inline npy_float imin(npy_int x, npy_int y) { return x > y ? y : x; }

inline npy_float imax(npy_int x, npy_int y) { return x > y ? x : y; }

/* SplitMix64 -- a small, high-quality PRNG (Steele et al. 2014; the seeder JAX/numpy
 * use). The whole RNG is JAX-style: a "key" is a uint64 state, advanced by next() for
 * a draw and SPLIT (split_key below) to derive an independent child stream for a
 * spawned secondary. There is no shared/global mutable seed -- each particle owns its
 * key, so the cascade is reproducible and event-parallel-safe. */
static inline uint64_t splitmix64_next(uint64_t *state) {
  uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/* Derive an independent child key from a parent key stream (JAX key split). */
static inline uint64_t split_key(uint64_t *state) { return splitmix64_next(state); }

/* Uniform float in [0, 1): top 24 bits of a SplitMix64 draw. */
static inline float rand01(uint64_t *state) {
  return (float)(splitmix64_next(state) >> 40) * (1.0f / 16777216.0f);
}

/* Per-solve() configuration + mutable dense output, threaded by pointer rather
 * than via file-scope statics: solve() runs under Py_BEGIN_ALLOW_THREADS, so
 * globals would be non-reentrant. Built once in solve(); passed (const) to
 * track_particle / dense_record, which only mutate the buffers it points at. */
/* Immutable simulation settings, constant for the whole solve() call. Per-step
 * dt is 0.9x the sagitta-bounded ceiling (chord-vs-arc error < straw resolution),
 * clamped to max_dt; a fixed `dt` (>0) overrides it. max_time caps total flight
 * time so trapped/curling tracks terminate. The trajectory buffer is sampled
 * evenly in time at step_t = max_time / n_traj (decoupled from the physics step). */
typedef struct {
  float max_dt, max_time, step_t;
  int n_traj;
  // Extra physics processes (all default off; baseline behaviour unchanged).
  float lambda_conv_cm;  // photon conversion length (cm); <=0 disables gamma->e+e-
  int enable_decay;      // charged pi/K decay-in-flight on/off
  float noise_rate;      // mean uncorrelated noise hits per event; <=0 disables
  // Secondary production in the thin straw wall, computed per hit from these
  // energy-INDEPENDENT material constants x the track's (z, beta, gamma):
  //   delta-ray rate ~ delta_const*(z^2/beta^2)*(1/Tcut - 1/Tmax)
  //   pair/brems     ~ 1 / lambda_conv_cm   (asymptotic, energy-independent)
  // delta_const = (K/2)(Z/A)rho [MeV/cm]; t_wall = straw wall thickness (cm, the
  // track crosses 2 per hit); delta_Tcut = delta-ray tracking threshold (MeV).
  float delta_const, t_wall, delta_Tcut;
  // Multiple-scattering material budget per layer crossing (x/X0, dimensionless);
  // <=0 disables the Highland kick (baseline clean tracks).
  float scatter_xX0;
  // Mean ionisation energy loss (Bethe-Bloch), applied once per wire-plane crossing
  // to range out soft secondaries (delta-rays). The per-crossing loss is
  //   dE = (eloss_wall_coef + eloss_gas_const * gas_path) * (z^2/beta^2) * Bethe_log
  // with eloss_wall_coef = K(Z/A)rho_wall * 2*t_wall [MeV] (fixed 2-wall path folded
  // in) and eloss_gas_const = K(Z/A)rho_gas [MeV/cm] (gas_path = slab chord, computed).
  // eloss_I = mean excitation energy (MeV) in the Bethe log; a track is stopped once
  // its kinetic energy drops below eloss_min_ke. enable_eloss == 0 disables it.
  int enable_eloss;
  float eloss_wall_coef, eloss_gas_const, eloss_I, eloss_min_ke;
  // Per-layer y-stagger (cm): straw centres in layer-in-view i are shifted by
  // -layer_y_offset/2 (i even) or +layer_y_offset/2 (i odd) -- the half-pitch
  // brick-laying that covers tube cusps and breaks left/right drift ambiguity.
  float layer_y_offset;
  // Integration caps / recursion / hit-buffer layout.
  npy_float dt_fixed;    // fixed-step override (>0); 0 -> adaptive per-step dt
  int n_steps;           // physics-loop safety cap (the loop is time-bounded)
  int max_depth, max_particles;
  int n_views, n_lpv, M; // decompose the global layer index + per-event hit cap
} SimConfig;

/* Per-event detector geometry (immutable during tracking). The layer arrays are
 * batch-wide and indexed by event via their strides; z0_field/B_sigma/B are this
 * event's field (peak z, Gaussian width, strength). */
typedef struct {
  const npy_float *layers, *angles;   // per-event design (z position + stereo angle per layer)
  npy_intp ls0, ls1, as0, as1;
  int n_layers, n_straws;
  npy_float z0_field, B_sigma, B;     // field: z0/B_sigma fixed (from Layout), B per-event design
  npy_float layer_width, layer_height;  // fixed straw half-length (x) / half-height (y), from Layout
} Geometry;

/* Mutable dense output, written in place: padded (n, M, 5) hits + (n, M) mask
 * (mask[e,i]=1 for a real hit, 0 for padding -- the per-event count is just
 * mask[e].sum()) and the (n, max_particles, n_traj, 3) trajectory viz buffer. */
typedef struct {
  npy_float *X;
  int *mask;
  // Optional per-hit TMCProcess code of the particle that caused the hit, parallel
  // to mask ((n, M) int32); NULL to skip. The first particle to fire a straw owns
  // the hit (later crossings of the same straw are deduped, so do not overwrite it).
  int *process_ids;
  npy_float *trajectories;
  npy_intp trs0, trs1, trs2, trs3;
  // Per-event per-straw scratch (FairShip-style dedup): tdc[k*n_straws + straw] is
  // the EARLIEST (min) TDC seen in that straw this event; an unfired straw holds the
  // STRAW_TDC_EMPTY sentinel (so a plain min() records). proc is the matching
  // causing-particle TMCProcess code. fired_idx lists the indices touched this event
  // (for O(n_fired) sparse-clear + the max_hits selection). Malloc'd once per solve()
  // and reused across events.
  npy_float *tdc;   // (n_layers * n_straws,) min TDC per straw; >= STRAW_TDC_EMPTY = unfired
  int *proc;        // (n_layers * n_straws,) process id of the min-TDC hit
  int *fired_idx;   // (n_layers * n_straws,) flat straw indices fired this event
  int n_fired;      // count of entries in fired_idx for the current event
  int n_straws;     // straws per layer (stride for the flat index)
  // Optional DEBUG MC-particle tree, flat/sparse across all events (NULL tree_int
  // to skip). One row per tracked particle (primary + secondary), appended via the
  // shared cursor *tree_count up to tree_cap; *tree_overflow is set if exceeded.
  //   tree_int  (cap, 4): [pdg, process_id, parent_id, n_hits]
  //   tree_float(cap, 7): [px, py, pz (MeV/c), x, y, z (cm, origin), t0 (ns)]
  //   tree_event(cap,)  : event index of the particle (CSR-style locator)
  int *tree_int;
  npy_float *tree_float;
  int *tree_event;
  int *tree_count;
  int tree_cap;
  int *tree_overflow;
} HitBuffers;

#define STRAW_TDC_EMPTY 1.0e30f  /* unfired-straw sentinel (so a plain min() records the first hit) */

/* Record one straw hit into the per-event scratch, keeping the EARLIEST (min) TDC
 * per straw (FairShip dedup: the tube latches the first pulse). k = global layer. */
static inline void record_hit(HitBuffers *out, int k, int straw_i, float fdigi, int process_id) {
  const npy_intp idx = (npy_intp)k * out->n_straws + straw_i;
  if (fdigi < out->tdc[idx]) {
    if (out->tdc[idx] >= STRAW_TDC_EMPTY) out->fired_idx[out->n_fired++] = (int)idx;  // first fire this event
    out->tdc[idx] = fdigi;
    out->proc[idx] = process_id;
  }
}

/* {tdc, flat-straw-index} pair + ascending-TDC comparator for the over-capacity sort. */
typedef struct { npy_float tdc; int idx; } TdcIdx;
static int cmp_tdc(const void *a, const void *b) {
  const npy_float ta = ((const TdcIdx *)a)->tdc, tb = ((const TdcIdx *)b)->tdc;
  return (ta > tb) - (ta < tb);
}

/* Emit one event's fired straws into the dense output: keep the cfg->M smallest-TDC
 * straws (drop the latest, like a per-event readout limit), decode each flat index ->
 * [station, view, layer, straw, tdc] (+ process id), then sparse-clear the scratch.
 * TDCs are emitted RELATIVE to the earliest hit (the event's min TDC is subtracted) --
 * the absolute trigger phase carries no information, so the event starts at its first
 * signal. `sel` is caller scratch of length >= n_fired for the over-capacity sort. */
static void emit_event(const SimConfig *cfg, HitBuffers *out, int event_idx, TdcIdx *sel) {
  const int per_station = cfg->n_views * cfg->n_lpv;
  const int n_fired = out->n_fired;
  if (n_fired == 0) return;
  // Earliest hit in the event = the time origin; subtract it from every emitted TDC.
  float min_tdc = out->tdc[out->fired_idx[0]];
  for (int i = 1; i < n_fired; ++i) {
    const float t = out->tdc[out->fired_idx[i]];
    if (t < min_tdc) min_tdc = t;
  }
  int n_emit = n_fired, sorted = 0;
  if (n_fired > cfg->M) {  // over capacity -> keep the M earliest-TDC straws
    for (int i = 0; i < n_fired; ++i) { sel[i].idx = out->fired_idx[i]; sel[i].tdc = out->tdc[out->fired_idx[i]]; }
    qsort(sel, n_fired, sizeof(TdcIdx), cmp_tdc);
    n_emit = cfg->M;
    sorted = 1;
  }
  for (int i = 0; i < n_emit; ++i) {
    const int idx = sorted ? sel[i].idx : out->fired_idx[i];
    const int k = idx / out->n_straws, straw_i = idx % out->n_straws;
    const int station = (per_station > 0) ? k / per_station : 0;
    const int rem = (per_station > 0) ? k % per_station : 0;
    const int view = (cfg->n_lpv > 0) ? rem / cfg->n_lpv : 0;
    const int lpv = (cfg->n_lpv > 0) ? rem % cfg->n_lpv : 0;
    const npy_intp base = (npy_intp)event_idx * cfg->M * 5 + (npy_intp)i * 5;
    out->X[base + 0] = (npy_float)station;
    out->X[base + 1] = (npy_float)view;
    out->X[base + 2] = (npy_float)lpv;
    out->X[base + 3] = (npy_float)straw_i;
    out->X[base + 4] = out->tdc[idx] - min_tdc;  // time relative to the first signal in the event
    out->mask[(npy_intp)event_idx * cfg->M + i] = 1;
    if (out->process_ids != NULL) out->process_ids[(npy_intp)event_idx * cfg->M + i] = out->proc[idx];
  }
  for (int i = 0; i < n_fired; ++i) out->tdc[out->fired_idx[i]] = STRAW_TDC_EMPTY;  // sparse-clear
  out->n_fired = 0;
}

/* Proper decay length c*tau (cm) for the chargeable, in-flight-decaying
 * species, keyed by mass (MeV). Returns 0 for stable / neutral / unknown
 * particles (no decay-in-flight applied). */
static inline float ctau_cm_for_mass(npy_float mass_mev) {
  if (f32_abs(mass_mev - 139.57f) < 1.0f) return 780.4f; /* pi+- */
  if (f32_abs(mass_mev - 493.68f) < 2.0f) return 371.2f; /* K+-  */
  return 0.0f;
}

/* FairShip-matched constants (see ../FairShip): speed of light and straw
 * digitisation. v_drift = 1/(30 ns/mm) and sigma_spatial come straight from
 * geometry_config.py; the TDC formula matches strawtubesHit.cxx. */
#define C_CM_PER_NS 29.9792458f
#define STRAW_VDRIFT 0.0033333333f   /* cm/ns */
#define STRAW_SIGMA_SPATIAL 0.012f   /* cm    */
#define MASS_E 0.511f                /* electron mass (MeV/c^2) */
#define MASS_MU 105.66f              /* muon mass (MeV/c^2) */
#define K_BORIS 44.937759f           /* 0.5e-9 * e[C] / (MeV/c^2 in kg); see track_particle */

/* Best-effort PDG code from our (mass, charge) tracking identity, for the debug
 * MC tree. Lepton PDG sign is opposite the charge (e- = 11, e+ = -11); meson /
 * baryon sign follows the charge. 0 for an unrecognised mass. */
static inline int pdg_from_mass_charge(npy_float mass, npy_float charge) {
  if (mass < SLOW) return 22;                                          /* photon  */
  if (f32_abs(mass - MASS_E) < 0.1f) return charge < 0 ? 11 : -11;     /* e-/e+   */
  if (f32_abs(mass - MASS_MU) < 1.0f) return charge < 0 ? 13 : -13;    /* mu-/mu+ */
  if (f32_abs(mass - 139.57f) < 1.5f) return charge > 0 ? 211 : -211;  /* pi+/-   */
  if (f32_abs(mass - 493.68f) < 2.0f) return charge > 0 ? 321 : -321;  /* K+/-    */
  if (f32_abs(mass - 938.27f) < 2.0f) return charge > 0 ? 2212 : -2212;/* p/pbar  */
  return 0;
}

/* Standard normal via Box-Muller (one of the pair). */
static inline float rand_normal(uint64_t *state, float mean, float sigma) {
  float u1 = rand01(state);
  float u2 = rand01(state);
  if (u1 < 1e-12f) u1 = 1e-12f;
  const float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
  return mean + sigma * z;
}

/* Highland multiple-scattering kick applied once per layer crossing. The track's
 * velocity direction is deflected by two independent Gaussian angular kicks (the
 * two transverse axes), each with RMS
 *   theta0 = (13.6 MeV / (beta * p)) * |q| * sqrt(x/X0) * [1 + 0.038 ln(x/X0)]
 * (PDG), with p the momentum magnitude (MeV/c) and x/X0 the layer material
 * budget. Speed |v| (= beta) is conserved -- scattering rotates the direction,
 * not its magnitude. Off when scatter_xX0 <= 0 (baseline behaviour unchanged). */
static inline void apply_scatter(uint64_t *state, npy_float xX0, npy_float charge,
                                 npy_float p_mag, npy_float beta,
                                 npy_float *vx, npy_float *vy, npy_float *vz) {
  if (xX0 <= 0.0f || beta <= SLOW || p_mag <= SLOW || f32_abs(charge) < SLOW) return;
  const float theta0 = (13.6f / (beta * p_mag)) * f32_abs(charge) *
                       sqrtf(xX0) * (1.0f + 0.038f * logf(xX0));
  if (!(theta0 > 0.0f)) return;
  const npy_float vmag = sqrtf((*vx) * (*vx) + (*vy) * (*vy) + (*vz) * (*vz));
  if (vmag <= SLOW) return;
  const npy_float ux = *vx / vmag, uy = *vy / vmag, uz = *vz / vmag;
  // Perpendicular basis: cross u with the axis it is least aligned with.
  npy_float ax = 0.0f, ay = 0.0f, az = 0.0f;
  if (f32_abs(ux) <= f32_abs(uy) && f32_abs(ux) <= f32_abs(uz)) ax = 1.0f;
  else if (f32_abs(uy) <= f32_abs(uz)) ay = 1.0f;
  else az = 1.0f;
  npy_float e1x = uy * az - uz * ay, e1y = uz * ax - ux * az, e1z = ux * ay - uy * ax;
  const npy_float e1n = sqrtf(e1x * e1x + e1y * e1y + e1z * e1z);
  if (e1n <= SLOW) return;
  e1x /= e1n; e1y /= e1n; e1z /= e1n;
  const npy_float e2x = uy * e1z - uz * e1y;  // u x e1 (already unit)
  const npy_float e2y = uz * e1x - ux * e1z;
  const npy_float e2z = ux * e1y - uy * e1x;
  const float t1 = rand_normal(state, 0.0f, theta0);
  const float t2 = rand_normal(state, 0.0f, theta0);
  npy_float nx = ux + t1 * e1x + t2 * e2x;
  npy_float ny = uy + t1 * e1y + t2 * e2y;
  npy_float nz = uz + t1 * e1z + t2 * e2z;
  const npy_float nn = sqrtf(nx * nx + ny * ny + nz * nz);
  if (nn <= SLOW) return;
  *vx = vmag * nx / nn;
  *vy = vmag * ny / nn;
  *vz = vmag * nz / nn;
}

/* Poisson sample (Knuth) -- fine for the small means used for noise. */
static inline int rand_poisson(uint64_t *state, float lam) {
  if (lam <= 0.0f) return 0;
  const float L = expf(-lam);
  float p = 1.0f;
  int k = 0;
  do {
    k++;
    p *= rand01(state);
  } while (p > L);
  return k - 1;
}

// Flat (T, SPACE_DIM) float32 vector pool for the sparse particle inputs.
const PyArrayObject *check_flat_vector(const PyObject *object, npy_intp total) {
  if (!PyArray_Check(object)) return NULL;
  const PyArrayObject *a = (const PyArrayObject *)object;
  if (PyArray_TYPE(a) == NPY_FLOAT32 && PyArray_NDIM(a) == 2 &&
      PyArray_DIM(a, 0) == total && PyArray_DIM(a, 1) == SPACE_DIM)
    return a;
  return NULL;
}

// Flat (T,) float32 scalar pool for the sparse particle inputs.
const PyArrayObject *check_flat_scalar(const PyObject *object, npy_intp total) {
  if (!PyArray_Check(object)) return NULL;
  const PyArrayObject *a = (const PyArrayObject *)object;
  if (PyArray_TYPE(a) == NPY_FLOAT32 && PyArray_NDIM(a) == 1 && PyArray_DIM(a, 0) == total)
    return a;
  return NULL;
}

const PyArrayObject *check_vector_array(const PyObject *object, int batch,
                                        int size) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject *array = (PyArrayObject *)object;

  if (
      // PyArray_IS_C_CONTIGUOUS(array) &&
      PyArray_TYPE(array) == NPY_FLOAT32 && PyArray_NDIM(array) == 3 &&
      PyArray_DIM(array, 0) == batch && PyArray_DIM(array, 1) == size &&
      PyArray_DIM(array, 2) == SPACE_DIM) {
    return array;
  } else {
    return NULL;
  }
}

const PyArrayObject *check_scalar_array(const PyObject *object, int batch,
                                        int size) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject *array = (PyArrayObject *)object;

  if (
      // PyArray_IS_C_CONTIGUOUS(array) &&
      PyArray_Check(array) && PyArray_TYPE(array) == NPY_FLOAT32 &&
      PyArray_NDIM(array) == 2 && PyArray_DIM(array, 0) == batch &&
      PyArray_DIM(array, 1) == size) {
    return array;
  } else {
    return NULL;
  }
}

const PyArrayObject *check_design_array(const PyObject *object, int batch) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject *array = (PyArrayObject *)object;

  if (
      // PyArray_IS_C_CONTIGUOUS(array) &&
      PyArray_Check(array) && PyArray_TYPE(array) == NPY_FLOAT32 &&
      PyArray_NDIM(array) == 1 && PyArray_DIM(array, 0) == batch) {
    return array;
  } else {
    return NULL;
  }
}

// Track a single particle through the detector + recursive secondary tracking
static void track_particle(
    const SimConfig *cfg,    // immutable simulation settings
    const Geometry *geom,    // this event's detector geometry
    HitBuffers *out,         // dense hit + trajectory output (mutated in place)
    int event_idx, int particle_idx,
    int parent_idx, int process_id,               // tree parent (-1 if primary) + TMCProcess code
    npy_float x0, npy_float y0, npy_float z0,     // initial position (cm)
    npy_float px0, npy_float py0, npy_float pz0,  // initial momentum (MeV/c)
    npy_float mass, npy_float charge,             // mass (MeV/c^2), charge (e)
    npy_float t_initial,                          // particle start time (ns)
    int start_step, int depth,                    // loop start index, recursion depth
    int *next_secondary_idx,                      // next free secondary slot (shared)
    uint64_t key                                  // this particle's JAX-style RNG key (split per secondary)
);

// Forward declaration for recursion
static void track_particle(
    const SimConfig *cfg, const Geometry *geom, HitBuffers *out,
    int event_idx, int particle_idx,
    int parent_idx, int process_id,
    npy_float x0, npy_float y0, npy_float z0,
    npy_float px0, npy_float py0, npy_float pz0,
    npy_float mass, npy_float charge,
    npy_float t_initial,
    int start_step, int depth,
    int *next_secondary_idx,
    uint64_t key
) {
  // Unpack the structs into the locals the physics body uses by bare name.
  const npy_float *layers = geom->layers, *angles = geom->angles;
  const npy_float layer_width = geom->layer_width, layer_height = geom->layer_height;  // fixed (from Layout)
  const npy_intp ls0 = geom->ls0, ls1 = geom->ls1, as0 = geom->as0, as1 = geom->as1;
  const int n_layers = geom->n_layers, n_straws = geom->n_straws;
  const npy_float z0_field = geom->z0_field, B_sigma = geom->B_sigma, B = geom->B;
  const npy_float dt_fixed = cfg->dt_fixed;  // fixed-step override (>0); 0 -> adaptive
  const int n_steps = cfg->n_steps, max_depth = cfg->max_depth, max_particles = cfg->max_particles;
  const float delta_const = cfg->delta_const, t_wall = cfg->t_wall, delta_Tcut = cfg->delta_Tcut;
  const float scatter_xX0 = cfg->scatter_xX0;
  const int enable_eloss = cfg->enable_eloss;
  const float eloss_wall_coef = cfg->eloss_wall_coef, eloss_gas_const = cfg->eloss_gas_const;
  const float eloss_I = cfg->eloss_I, eloss_min_ke = cfg->eloss_min_ke;
  const float max_dt = cfg->max_dt, max_time = cfg->max_time, step_t = cfg->step_t, lambda_conv_cm = cfg->lambda_conv_cm;
  const int n_traj = cfg->n_traj, enable_decay = cfg->enable_decay;
  npy_float *trajectories = out->trajectories;
  const npy_intp trs0 = out->trs0, trs1 = out->trs1, trs2 = out->trs2, trs3 = out->trs3;

  // This particle's RNG stream is seeded from its own key (JAX-style); draws advance
  // rng_state, and each spawned secondary gets an independent stream via split_key().
  uint64_t rng_state = key;

  npy_float x = x0;
  npy_float y = y0;
  npy_float z = z0;

  npy_float px = px0;
  npy_float py = py0;
  npy_float pz = pz0;

  // printf("track_particle: event=%d, particle=%d, depth=%d, pos=(%.2f,%.2f,%.2f), p=(%.2f,%.2f,%.2f), mass=%.2f, charge=%.2f\n",
  //        event_idx, particle_idx, depth, x, y, z, px, py, pz, mass, charge);

  // Empty slot / zero-momentum ghost: nothing to track.
  if (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW) {
    return;
  }

  // Debug MC tree: append one row for this (real) particle. n_hits is filled in at
  // each exit via TREE_NHITS(). tree_slot < 0 means tree disabled or capacity hit.
  int tree_slot = -1;
  int n_hits_local = 0;
  if (out->tree_int != NULL) {
    const int slot = (*out->tree_count)++;
    if (slot >= out->tree_cap) {
      *out->tree_overflow = 1;
    } else {
      tree_slot = slot;
      out->tree_int[(npy_intp)slot * 4 + 0] = pdg_from_mass_charge(mass, charge);
      out->tree_int[(npy_intp)slot * 4 + 1] = process_id;
      out->tree_int[(npy_intp)slot * 4 + 2] = parent_idx;
      out->tree_int[(npy_intp)slot * 4 + 3] = 0;  // n_hits, finalised at exit
      out->tree_float[(npy_intp)slot * 7 + 0] = px0;
      out->tree_float[(npy_intp)slot * 7 + 1] = py0;
      out->tree_float[(npy_intp)slot * 7 + 2] = pz0;
      out->tree_float[(npy_intp)slot * 7 + 3] = x0;
      out->tree_float[(npy_intp)slot * 7 + 4] = y0;
      out->tree_float[(npy_intp)slot * 7 + 5] = z0;
      out->tree_float[(npy_intp)slot * 7 + 6] = t_initial;
      out->tree_event[slot] = event_idx;
    }
  }
#define TREE_NHITS() do { if (tree_slot >= 0) out->tree_int[(npy_intp)tree_slot * 4 + 3] = n_hits_local; } while (0)

  // Photon (massless): no ionisation, so it is invisible to the straws unless
  // it converts to an e+e- pair (gamma -> e+ e-). It records no hits itself;
  // the conversion point is sampled exponentially along its straight path and
  // the pair (opposite charges, half the energy each) is tracked from there.
  if (mass < SLOW) {
    if (lambda_conv_cm > 0.0f && depth < max_depth) {
      const npy_float p_mag = sqrtf(px * px + py * py + pz * pz);
      const npy_float ux = px / p_mag, uy = py / p_mag, uz = pz / p_mag;
      const npy_float dl = max_dt * C_CM_PER_NS; /* straight photon step (cm), speed c */
      const float pconv = 1.0f - expf(-dl / lambda_conv_cm);
      npy_float t_ph = t_initial;
      for (int j = start_step; j < n_steps && (t_ph - t_initial) < max_time; ++j) {
        if (rand01(&rng_state) < pconv) {
          {  // faithful cascade: always spawn (max_particles no longer gates hits)
            const int e_minus = (*next_secondary_idx)++;
            const int e_plus = (*next_secondary_idx)++;
            const npy_float pe = 0.5f * p_mag; /* split photon energy */
            const npy_float charges_pair[2] = {-1.0f, +1.0f};
            const int idx_pair[2] = {e_minus, e_plus};
            for (int s = 0; s < 2; ++s) {
              track_particle(
                  cfg, geom, out,
                  event_idx, idx_pair[s], particle_idx, PROC_PAIR,
                  x, y, z, ux * pe, uy * pe, uz * pe,
                  MASS_E, charges_pair[s], t_ph, 0, depth + 1,
                  next_secondary_idx, split_key(&rng_state));
            }
          }
          TREE_NHITS();
          return; /* photon consumed at conversion */
        }
        x += ux * dl; /* propagate straight to the next step */
        y += uy * dl;
        z += uz * dl;
        t_ph += max_dt;
      }
    }
    TREE_NHITS();
    return; /* no conversion (or disabled): photon leaves no hits */
  }

  // printf("  -> Valid particle, tracking through %d steps, start_step=%d, n_steps=%d\n", n_steps - start_step, start_step, n_steps);
  // printf("  -> z0_field=%.2f, B_sigma=%.2f, n_layers=%d, n_straws=%d\n", z0_field, B_sigma, n_layers, n_straws);

  npy_float p2 = px * px + py * py + pz * pz;
  npy_float p_mag = sqrtf(p2);  // |p| (MeV/c); conserved under static B, decremented by energy loss
  npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));
  // printf("  -> gamma=%.5f\n", gamma);
  npy_float vx = px / (gamma * mass); // v's are dimensionless, in units of c
  npy_float vy = py / (gamma * mass);
  npy_float vz = pz / (gamma * mass);
  //printf("  -> v=%.5f\n", sqrtf(vx * vx + vy * vy + vz * vz));

  // The per-step dt is chosen adaptively in the loop (sagitta-bounded). The
  // Bx-independent part of that bound is constant along the path -- |p|, gamma,
  // beta are conserved in a static B field -- so precompute it once:
  //   dt_sag = 0.9 * sqrt(sag_num / Bx),  sag_num = 4*sigma*m*gamma/(beta*c*K*|q|)
  // Derivation: the Boris push rotates v by theta with tan(theta/2) =
  // K_BORIS*dt*q*Bx/(m*gamma) (line below); the chord is L = beta*C_CM*dt; the
  // chord-vs-arc sagitta s = L*tan(theta/4)/2 ~ L*theta/8. Imposing s <= sigma
  // (STRAW_SIGMA_SPATIAL) and solving for dt gives the bound above.
  npy_float beta = sqrtf(vx * vx + vy * vy + vz * vz);
  const npy_float qabs = f32_abs(charge);
  npy_float sag_num =
      (qabs > SLOW && beta > SLOW)
          ? 4.0f * STRAW_SIGMA_SPATIAL * mass * gamma / (beta * C_CM_PER_NS * K_BORIS * qabs)
          : 1e30f;  // straight / neutral track -> field imposes no ceiling (recomputed on energy loss)

  // Per-layer geometry is invariant across steps -> precompute once. -O3 can't
  // hoist it from the step loop because the layer arrays may alias the dense
  // output buffers written each step; doing it by hand removes the redundancy
  // (was the dominant cost in the callgrind profile).
  npy_float pl_z[n_layers], pl_height[n_layers], pl_width[n_layers];
  npy_float pl_r[n_layers], pl_left[n_layers], pl_right[n_layers], pl_yoff[n_layers];
  int order[n_layers];
  npy_float max_right = -1e30f;
  // Half-pitch stagger: even layer-in-view -> -h, odd -> +h. per_station decomposes
  // the global layer index k into (station, view, layer-in-view) like dense_record.
  const npy_float yoff_half = 0.5f * cfg->layer_y_offset;
  const int per_station = cfg->n_views * cfg->n_lpv;
  const npy_float lr = layer_height / n_straws;  // straw radius (height is half-height); fixed
  for (int k = 0; k < n_layers; ++k) {
    const npy_float lz = layers[event_idx * ls0 + k * ls1];
    const int lpv = (cfg->n_lpv > 0) ? (k % per_station) % cfg->n_lpv : 0;
    pl_z[k] = lz;
    pl_height[k] = layer_height;
    pl_width[k] = layer_width;
    pl_r[k] = lr;
    pl_left[k] = lz - lr;  // layer_half_thickness = r
    pl_right[k] = lz + lr;
    pl_yoff[k] = (lpv & 1) ? yoff_half : -yoff_half;  // even i -> -h, odd i -> +h
    if (pl_right[k] > max_right) max_right = pl_right[k];
    order[k] = k;
  }
  // Sort layer indices by z (insertion sort; n_layers is small). The optimiser
  // may place stations out of z-order, so don't assume the input is sorted.
  for (int a = 1; a < n_layers; ++a) {
    const int key = order[a];
    const npy_float kz = pl_z[key];
    int b = a - 1;
    while (b >= 0 && pl_z[order[b]] > kz) {
      order[b + 1] = order[b];
      b--;
    }
    order[b + 1] = key;
  }

  npy_float t_now = t_initial;
  int last_traj = 0;
  for (int j = start_step; j < n_steps; ++j) {
    if (t_now - t_initial >= max_time) break;  // trapped/curling -> give up

    const npy_float Bx = B * exp(-0.5f * square((z - z0_field) / B_sigma));  // proper Gaussian: B_sigma is the std-dev
    const npy_float absBx = f32_abs(Bx);  // sagitta bound depends on |Bx|; sign is kept in the push below

    // Adaptive step: 0.9x the sagitta ceiling, clamped to max_dt. A fixed dt
    // (dt_fixed > 0) overrides the adaptive choice.
    npy_float dt_step = max_dt;
    if (dt_fixed > 0.0f) {
      dt_step = dt_fixed;
    } else if (absBx > SLOW) {
      const npy_float dt_sag = 0.9f * sqrtf(sag_num / absBx);
      if (dt_sag < dt_step) dt_step = dt_sag;
    }

    const npy_float c = K_BORIS * dt_step * charge / (mass * gamma);
    const npy_float tx = c * Bx;
    const npy_float t_norm_sqr = tx * tx;

    // Boris pusher: v- + (v- x t) -> v', then v+ = v- + v' x s. The final cross
    // product adds onto the ORIGINAL v- (not v'); composing onto v' would scale
    // |v| by (1-sx*tx)^2 + (tx+sx)^2 != 1 (energy gain). a^2 + sx^2 = 1 here.
    const npy_float vy_m = vy + vz * tx;
    const npy_float vz_m = vz - vy * tx;

    const npy_float sx = 2 * tx / (1 + t_norm_sqr);

    vy = vy + vz_m * sx;
    vz = vz - vy_m * sx;

    const npy_float dx = dt_step * vx * C_CM_PER_NS;
    const npy_float dy = dt_step * vy * C_CM_PER_NS;
    const npy_float dz = dt_step * vz * C_CM_PER_NS;

    const npy_float x_ = x + dx;
    const npy_float y_ = y + dy;
    const npy_float z_ = z + dz;
    // Debug
    //if (j % 50 == 0) printf("  Step %d: pos=(%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f), time %f\n", j, x, y, z, x_, y_, z_, j * dt);


    // Scan only the layers the step [zmin, zmax] can reach, in z-order. Binary
    // search the first layer whose right edge reaches zmin, then scan until a
    // left edge passes zmax. No persistent cursor -> correct in either direction
    // (handles curling tracks). Assumes a uniform straw radius (true for the SST)
    // so pl_right is sorted in `order`.
    const npy_float zmin = (z < z_) ? z : z_;
    const npy_float zmax = (z > z_) ? z : z_;
    int klo = 0, khi = n_layers;
    while (klo < khi) {
      const int mid = (klo + khi) >> 1;
      if (pl_right[order[mid]] < zmin)
        klo = mid + 1;
      else
        khi = mid;
    }
    for (int ki = klo; ki < n_layers; ++ki) {
      const int k = order[ki];
      if (pl_left[k] > zmax) break;  // ahead -> all later (sorted) layers too
      const npy_float layer = pl_z[k];
      const npy_float height = pl_height[k];
      const npy_float width = pl_width[k];
      const npy_float r = pl_r[k];
      const npy_float yoff = pl_yoff[k];  // half-pitch stagger of this layer's straw centres
      // if (event_idx == 0 && particle_idx < 2 && k == 8) {
      //   printf("  -> CROSSING layer %d at step %d!\n", k, j);
      // }

      const npy_float angle = angles[event_idx * as0 + k * as1];
      const npy_float ca = cosf(angle), sa = sinf(angle);
      const npy_float ta = sa / ca;  // tan(angle); ca > 0 for the bounded stereo angles

      // Layer is a PARALLELOGRAM centred at (0,0,layer): vertical sides at x=+/-width
      // (parallel to y), straws stacked in y at pitch 2r. Each straw (tube + its central
      // sense wire) is tilted by `angle` from x, ends locked to the x=+/-width sides. The
      // sheared coordinate Y = y - x*tan(angle) is constant (= the straw centre y_i) along
      // a straw, so the frame is |x|<=width AND |Y|<=height, and the index inverts from Y.
      const npy_float Y0 = y - x * ta;
      if (f32_abs(x) > width || f32_abs(Y0) > height) {
        continue;  // outside the parallelogram frame
      }
      const npy_float Y1 = y_ - x_ * ta;

      // The segment spans Y in [Ymin, Ymax]; convert to a straw index RANGE (a track
      // crossing at an angle passes through several adjacent tubes); the +/-1 pad
      // covers the straw-radius overlap. EVERY tube actually entered (dist < r) fires.
      const npy_float Ymin = (Y0 < Y1) ? Y0 : Y1;
      const npy_float Ymax = (Y0 > Y1) ? Y0 : Y1;
      // Invert straw_y = (2i+1)*r - height + yoff -> i = floor((Y + height - yoff)/(2r)).
      npy_int i_lo = (npy_int)floorf(0.5f * (Ymin + height - yoff) / r) - 1;
      npy_int i_hi = (npy_int)floorf(0.5f * (Ymax + height - yoff) / r) + 1;
      if (i_lo < 0) i_lo = 0;
      if (i_hi >= n_straws) i_hi = n_straws - 1;

      // 3D distance from the track segment to each tilted wire (closest approach of two
      // lines). The wire direction d_w = (ca, sa, 0) is shared by the layer, so the
      // cross product (d_t x d_w) and its norm are hoisted; only the wire centre y_i
      // varies per straw, entering the dot product linearly via `base`.
      const npy_float dtx = x_ - x, dty = y_ - y, dtz = z_ - z;
      const npy_float cx = -dtz * sa, cy = dtz * ca, cz = dtx * sa - dty * ca;
      const npy_float cross_norm2 = cx * cx + cy * cy + cz * cz;
      const npy_float base = -x * cx - y * cy + (layer - z) * cz;  // (W_i-P0).(d_t x d_w) = base + y_i*cy

      int fired_any = 0;
      for (npy_int straw_i = i_lo; straw_i <= i_hi; ++straw_i) {
        const npy_float straw_y = (2 * straw_i + 1) * r - height + yoff;  // wire centre y (at x=0)

        npy_float sqr_distance_to_wire;
        if (cross_norm2 < 1e-12f) {
          // Track parallel to the wire: perpendicular point-to-line distance from P0.
          const npy_float vx_ = -x, vy_ = straw_y - y, vz_ = layer - z;
          const npy_float proj = vx_ * ca + vy_ * sa;  // (W_i-P0).d_w, d_w unit
          sqr_distance_to_wire = vx_ * vx_ + vy_ * vy_ + vz_ * vz_ - proj * proj;
        } else {
          const npy_float dot = base + straw_y * cy;
          sqr_distance_to_wire = dot * dot / cross_norm2;
        }

        if (sqr_distance_to_wire >= r * r) continue;  // tube not entered -> no hit

        // FairShip TDC: t_MC + |Gaus(dist,sigma)|/v_drift + propagation along the wire.
        // The +x readout end is at x=width; the along-wire distance from the hit (at x)
        // is (width-x)/cos(angle) (arc length per unit x is 1/cos).
        const float dist_cm = sqrtf(sqr_distance_to_wire);
        const float t_drift = fabsf(rand_normal(&rng_state, dist_cm, STRAW_SIGMA_SPATIAL)) / STRAW_VDRIFT;
        const float t_prop = (width - x) / (ca * C_CM_PER_NS);
        const float fdigi = t_now + t_drift + t_prop;
        record_hit(out, k, straw_i, fdigi, process_id);  // min-TDC dedup into per-straw scratch
        fired_any = 1;
        n_hits_local++;
      }

      // Highland multiple scattering: one kick per layer crossing the track passes
      // through (it is inside the frame here). Deflects v for the onward steps;
      // off when scatter_xX0 <= 0.
      apply_scatter(&rng_state, scatter_xX0, charge, p_mag, beta, &vx, &vy, &vz);

      // Mean ionisation energy loss (Bethe-Bloch), once per wire-plane crossing -- gated
      // on the z=layer straddle so a slow track taking many small steps across the 2r
      // slab is charged exactly once. Walls: fixed 2*t_wall (folded into eloss_wall_coef).
      // Gas: the slab chord 2r/|cos z| = 2r*beta/|vz| (grazing/curling tracks eat more).
      // Update |p|/gamma/beta + the sagitta-dt coeff; stop the track once it ranges out.
      if (enable_eloss && qabs > SLOW && beta > SLOW && (z - layer) * (z_ - layer) <= 0.0f) {
        const npy_float vzc = f32_abs(vz);
        npy_float gas_path = (vzc > 1.0e-4f) ? 2.0f * r * beta / vzc : 2.0e4f * r;
        if (gas_path > 200.0f) gas_path = 200.0f;  // cap a near-tangent crossing (~straw length)
        const float b2 = beta * beta;
        const float me_over_M = MASS_E / mass;
        const float Tmax = 2.0f * MASS_E * b2 * gamma * gamma /
                           (1.0f + 2.0f * gamma * me_over_M + me_over_M * me_over_M);
        float Ln = 0.5f * logf(2.0f * MASS_E * b2 * gamma * gamma * Tmax / (eloss_I * eloss_I)) - b2;
        if (Ln < 0.0f) Ln = 0.0f;
        const float dE = (eloss_wall_coef + eloss_gas_const * gas_path) * (charge * charge / b2) * Ln;
        const float E_new = gamma * mass - dE;
        if (E_new <= mass + eloss_min_ke) {  // ranged out -> stop the track here
          TREE_NHITS();
          return;
        }
        gamma = E_new / mass;
        p_mag = sqrtf(E_new * E_new - mass * mass);
        p2 = p_mag * p_mag;
        const npy_float beta_new = p_mag / E_new;
        const npy_float vscale = beta_new / beta;
        vx *= vscale;
        vy *= vscale;
        vz *= vscale;
        beta = beta_new;
        sag_num = (qabs > SLOW && beta > SLOW)
                      ? 4.0f * STRAW_SIGMA_SPATIAL * mass * gamma / (beta * C_CM_PER_NS * K_BORIS * qabs)
                      : 1e30f;
      }

      if (!fired_any) {
        continue;  // no tube entered (or all already fired / buffer full) -> no secondaries
      }

      // Per-layer secondary production in the thin straw wall (track crosses 2
      // walls per hit). Probabilities are built here from energy-independent
      // material constants x the track kinematics:
      //   delta-ray:  p = delta_const * L * (z^2/beta^2) * (1/Tcut - 1/Tmax)
      //   pair/brems: p = L / lambda_conv_cm        (asymptotic, energy-indep.)
      if (depth < max_depth) {
        const float L_wall = 2.0f * t_wall;
        const float me_over_M = MASS_E / mass;
        const float Tmax = 2.0f * MASS_E * beta * beta * gamma * gamma /
                           (1.0f + 2.0f * gamma * me_over_M + me_over_M * me_over_M);
        float p_single = 0.0f;
        if (beta > SLOW && delta_Tcut > 0.0f && Tmax > delta_Tcut) {
          p_single = delta_const * L_wall * (charge * charge) / (beta * beta) *
                     (1.0f / delta_Tcut - 1.0f / Tmax);
        }
        const float p_pair = (lambda_conv_cm > 0.0f) ? L_wall / lambda_conv_cm : 0.0f;

        if (p_single > 0.0f || p_pair > 0.0f) {
          const float r_spawn = rand01(&rng_state);
          if (r_spawn < p_single) {
            // delta-ray: energy ~ 1/T^2 on [Tcut, Tmax] (inverse-CDF), isotropic.
            const int e_idx = (*next_secondary_idx)++;
            const float uu = rand01(&rng_state);
            const float T = 1.0f / (1.0f / delta_Tcut - uu * (1.0f / delta_Tcut - 1.0f / Tmax));
            const float u = rand01(&rng_state), v = rand01(&rng_state);
            const float cos_theta = 2.0f * u - 1.0f;
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
            const float phi = 2.0f * (float)M_PI * v;
            const npy_float p_sec = sqrtf(T * T + 2.0f * T * MASS_E);
            track_particle(cfg, geom, out, event_idx, e_idx, particle_idx, PROC_DELTARAY, x_, y_, z_,
                           p_sec * sin_theta * cosf(phi), p_sec * sin_theta * sinf(phi),
                           p_sec * cos_theta, MASS_E, -1.0f, t_now, 0, depth + 1,
                           next_secondary_idx, split_key(&rng_state));
          } else if (r_spawn < p_single + p_pair) {
            // e+e- pair from a hard radiated photon: 1/k brems spectrum (log-uniform
            // on [Tcut, KE]), each lepton gets half, emitted back-to-back.
            const int e_minus_idx = (*next_secondary_idx)++;
            const int e_plus_idx = (*next_secondary_idx)++;
            const float KE = fmaxf((gamma - 1.0f) * mass, delta_Tcut * 1.001f);
            const float kgamma = delta_Tcut * expf(rand01(&rng_state) * logf(KE / delta_Tcut));
            const float T_half = 0.5f * kgamma;
            const float u = rand01(&rng_state), v = rand01(&rng_state);
            const float cos_theta = 2.0f * u - 1.0f;
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
            const float phi = 2.0f * (float)M_PI * v;
            const npy_float p_sec = sqrtf(T_half * T_half + 2.0f * T_half * MASS_E);
            const npy_float pxm = p_sec * sin_theta * cosf(phi);
            const npy_float pym = p_sec * sin_theta * sinf(phi);
            const npy_float pzm = p_sec * cos_theta;
            track_particle(cfg, geom, out, event_idx, e_minus_idx, particle_idx, PROC_PAIR, x_, y_, z_,
                           pxm, pym, pzm, MASS_E, -1.0f, t_now, 0, depth + 1, next_secondary_idx, split_key(&rng_state));
            track_particle(cfg, geom, out, event_idx, e_plus_idx, particle_idx, PROC_PAIR, x_, y_, z_,
                           -pxm, -pym, -pzm, MASS_E, +1.0f, t_now, 0, depth + 1, next_secondary_idx, split_key(&rng_state));
          }
        }
      }
    }

    x = x_;
    y = y_;
    z = z_;
    t_now += dt_step;

    // Trajectory: sampled evenly in elapsed time (viz only). One physics step may
    // span many sample slots (super-sample) or none (sub-sample); fill every slot
    // whose timestamp i*step_t the accumulated time has now passed.
    if (trajectories != NULL && particle_idx < max_particles && step_t > 0.0f) {
      int upto = (int)((t_now - t_initial) / step_t);
      if (upto > n_traj) upto = n_traj;
      const npy_intp tbase = event_idx * trs0 + particle_idx * trs1;
      for (int i = last_traj; i < upto; ++i) {
        trajectories[tbase + i * trs2] = x;
        trajectories[tbase + i * trs2 + trs3] = y;
        trajectories[tbase + i * trs2 + 2 * trs3] = z;
      }
      last_traj = upto;
    }

    // Past the last layer and still moving forward -> no more hits possible.
    if (vz > 0.0f && z > max_right) break;

    // Decay-in-flight: pi+- / K+- -> mu + nu. The muon carries the bulk of the
    // momentum and continues nearly collinearly (a small kink); the parent
    // stops here. Decay probability over this step uses the lab decay length
    // L = beta*gamma*c*tau, so dl/L = c*dt / (gamma*c*tau).
    if (enable_decay && depth < max_depth) {
      const float ctau = ctau_cm_for_mass(mass);
      if (ctau > 0.0f) {
        const float pdecay = 1.0f - expf(-dt_step * C_CM_PER_NS / (gamma * ctau));
        if (rand01(&rng_state) < pdecay) {
          const int mu_idx = (*next_secondary_idx)++;
          // Muon direction = current velocity direction; |p| conserved by B.
          const npy_float pmag = sqrtf(px * px + py * py + pz * pz);
          const npy_float vmag = sqrtf(vx * vx + vy * vy + vz * vz);
          const npy_float scale = (vmag > SLOW) ? pmag / vmag : 0.0f;
          track_particle(
              cfg, geom, out,
              event_idx, mu_idx, particle_idx, PROC_DECAY,
              x, y, z, vx * scale, vy * scale, vz * scale,
              MASS_MU, charge, t_now, 0, depth + 1,
              next_secondary_idx, split_key(&rng_state));  // t_now = decay time; muon starts fresh
          TREE_NHITS();
          return; /* parent stops at the decay point */
        }
      }
    }
  }
  TREE_NHITS();
#undef TREE_NHITS
}

/* ====================== Init-once Python/C-API objects ======================
 * The constant solver inputs are bundled into extension types built once and reused:
 *   SimParams    - physics + solver constants
 *   Layout       - fixed structural counts
 *   InputEvents  - the validated particle pool (holds refs to its numpy arrays)
 *   DebugBuffers - optional debug outputs (trajectories / process_ids / MC tree)
 * solve() reads these plus the per-call explicit arrays (seeds, boundaries, design,
 * scratch, X/mask/counts). Any type that stores a numpy data pointer Py_INCREFs the
 * backing array in tp_init and Py_XDECREFs it in tp_dealloc. */

/* ---- SimParams: physics + solver constants ---- */
// Embeds the SimConfig the tracker consumes directly: init fills the physics +
// solver fields once. solve() copies the struct and overlays the fields that come
// from elsewhere -- Layout (n_views/n_lpv/M/layer_y_offset) and the per-call debug
// buffers (max_particles/n_traj/step_t) -- so there is no field-by-field repackaging.
typedef struct {
  PyObject_HEAD
  SimConfig cfg;
} SimParamsObject;

static int SimParams_init(SimParamsObject *s, PyObject *args, PyObject *kwds) {
  (void)kwds;
  return PyArg_ParseTuple(args, "fffiifffifffiffff", &s->cfg.max_dt, &s->cfg.max_time, &s->cfg.dt_fixed,
                          &s->cfg.n_steps, &s->cfg.max_depth, &s->cfg.scatter_xX0, &s->cfg.lambda_conv_cm,
                          &s->cfg.noise_rate, &s->cfg.enable_decay, &s->cfg.delta_const, &s->cfg.t_wall,
                          &s->cfg.delta_Tcut, &s->cfg.enable_eloss, &s->cfg.eloss_wall_coef,
                          &s->cfg.eloss_gas_const, &s->cfg.eloss_I, &s->cfg.eloss_min_ke)
             ? 0
             : -1;
}
static PyTypeObject SimParamsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "straw_detector.SimParams",
    .tp_basicsize = sizeof(SimParamsObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SimParams_init,
};

/* ---- Layout: fixed structural counts + fixed geometry/field constants ---- */
typedef struct {
  PyObject_HEAD
  int n_views, n_lpv, n_straws, n_layers, M;  // M = max_hits_per_event
  float layer_y_offset, layer_width, layer_height;  // straw half-length(x) / half-height(y)
  float z0, B_sigma;  // magnet centre (cm) + on-axis Gaussian field width (cm) -- not optimised
} LayoutObject;

static int Layout_init(LayoutObject *s, PyObject *args, PyObject *kwds) {
  (void)kwds;
  return PyArg_ParseTuple(args, "iiiiifffff", &s->n_views, &s->n_lpv, &s->n_straws, &s->n_layers,
                          &s->M, &s->layer_y_offset, &s->layer_width, &s->layer_height, &s->z0, &s->B_sigma)
             ? 0
             : -1;
}
static PyTypeObject LayoutType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "straw_detector.Layout",
    .tp_basicsize = sizeof(LayoutObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Layout_init,
};

/* ---- Scratch: the reused per-straw sparse-set buffers (tdc/proc/fired_idx) ---- */
typedef struct {
  PyObject_HEAD
  PyObject *o_tdc, *o_proc, *o_fired_idx;  // INCREF'd backing arrays
  npy_float *tdc; int *proc; int *fired_idx;
  npy_intp n_cells;
} ScratchObject;

static int Scratch_init(ScratchObject *s, PyObject *args, PyObject *kwds) {
  (void)kwds;
  PyObject *ot, *op, *of;
  if (!PyArg_ParseTuple(args, "OOO", &ot, &op, &of)) return -1;
  if (!PyArray_Check(ot) || PyArray_TYPE((PyArrayObject *)ot) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "tdc must be a float32 array");
    return -1;
  }
  const npy_intp n = PyArray_SIZE((PyArrayObject *)ot);
  if (!PyArray_Check(op) || PyArray_TYPE((PyArrayObject *)op) != NPY_INT32 || PyArray_SIZE((PyArrayObject *)op) != n ||
      !PyArray_Check(of) || PyArray_TYPE((PyArrayObject *)of) != NPY_INT32 || PyArray_SIZE((PyArrayObject *)of) != n) {
    PyErr_SetString(PyExc_TypeError, "proc and fired_idx must be int32 arrays the same size as tdc");
    return -1;
  }
  s->n_cells = n;
  s->tdc = (npy_float *)PyArray_DATA((PyArrayObject *)ot);
  s->proc = (int *)PyArray_DATA((PyArrayObject *)op);
  s->fired_idx = (int *)PyArray_DATA((PyArrayObject *)of);
  Py_INCREF(ot); s->o_tdc = ot;
  Py_INCREF(op); s->o_proc = op;
  Py_INCREF(of); s->o_fired_idx = of;
  return 0;
}
static void Scratch_dealloc(ScratchObject *s) {
  Py_XDECREF(s->o_tdc); Py_XDECREF(s->o_proc); Py_XDECREF(s->o_fired_idx);
  Py_TYPE(s)->tp_free((PyObject *)s);
}
static PyTypeObject ScratchType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "straw_detector.Scratch",
    .tp_basicsize = sizeof(ScratchObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Scratch_init,
    .tp_dealloc = (destructor)Scratch_dealloc,
};

/* ---- InputEvents: the validated, ref-held particle pool ---- */
typedef struct {
  PyObject_HEAD
  PyObject *o_mass, *o_chg, *o_pos, *o_mom, *o_tim;  // INCREF'd backing arrays
  const npy_float *mass, *chg, *pos, *mom, *tim;     // data pointers
  npy_intp T;                                        // pool length
  npy_intp ms0, chs0, its0, ips0, ips1, ivs0, ivs1;  // strides (in elements)
} InputEventsObject;

static int IE_is_1d(PyObject *o, npy_intp T) {
  return PyArray_Check(o) && PyArray_TYPE((PyArrayObject *)o) == NPY_FLOAT32 &&
         PyArray_NDIM((PyArrayObject *)o) == 1 && PyArray_DIM((PyArrayObject *)o, 0) == T;
}
static int IE_is_2d(PyObject *o, npy_intp T) {
  return PyArray_Check(o) && PyArray_TYPE((PyArrayObject *)o) == NPY_FLOAT32 &&
         PyArray_NDIM((PyArrayObject *)o) == 2 && PyArray_DIM((PyArrayObject *)o, 0) == T &&
         PyArray_DIM((PyArrayObject *)o, 1) == SPACE_DIM;
}
static int InputEvents_init(InputEventsObject *s, PyObject *args, PyObject *kwds) {
  (void)kwds;
  PyObject *om, *oc, *op, *ov, *ot;
  if (!PyArg_ParseTuple(args, "OOOOO", &om, &oc, &op, &ov, &ot)) return -1;
  if (!PyArray_Check(om) || PyArray_TYPE((PyArrayObject *)om) != NPY_FLOAT32 ||
      PyArray_NDIM((PyArrayObject *)om) != 1) {
    PyErr_SetString(PyExc_TypeError, "masses must be a (T,) float32 array");
    return -1;
  }
  const npy_intp T = PyArray_DIM((PyArrayObject *)om, 0);
  if (!IE_is_1d(oc, T) || !IE_is_1d(ot, T)) {
    PyErr_SetString(PyExc_TypeError, "charges and times must be (T,) float32 arrays");
    return -1;
  }
  if (!IE_is_2d(op, T) || !IE_is_2d(ov, T)) {
    PyErr_SetString(PyExc_TypeError, "positions and momenta must be (T, 3) float32 arrays");
    return -1;
  }
  s->T = T;
  s->mass = (const npy_float *)PyArray_DATA((PyArrayObject *)om);
  s->chg = (const npy_float *)PyArray_DATA((PyArrayObject *)oc);
  s->tim = (const npy_float *)PyArray_DATA((PyArrayObject *)ot);
  s->pos = (const npy_float *)PyArray_DATA((PyArrayObject *)op);
  s->mom = (const npy_float *)PyArray_DATA((PyArrayObject *)ov);
  s->ms0 = PyArray_STRIDE((PyArrayObject *)om, 0) / sizeof(npy_float);
  s->chs0 = PyArray_STRIDE((PyArrayObject *)oc, 0) / sizeof(npy_float);
  s->its0 = PyArray_STRIDE((PyArrayObject *)ot, 0) / sizeof(npy_float);
  s->ips0 = PyArray_STRIDE((PyArrayObject *)op, 0) / sizeof(npy_float);
  s->ips1 = PyArray_STRIDE((PyArrayObject *)op, 1) / sizeof(npy_float);
  s->ivs0 = PyArray_STRIDE((PyArrayObject *)ov, 0) / sizeof(npy_float);
  s->ivs1 = PyArray_STRIDE((PyArrayObject *)ov, 1) / sizeof(npy_float);
  Py_INCREF(om); s->o_mass = om;  // keep arrays alive while we hold their data pointers
  Py_INCREF(oc); s->o_chg = oc;
  Py_INCREF(ot); s->o_tim = ot;
  Py_INCREF(op); s->o_pos = op;
  Py_INCREF(ov); s->o_mom = ov;
  return 0;
}
static void InputEvents_dealloc(InputEventsObject *s) {
  Py_XDECREF(s->o_mass); Py_XDECREF(s->o_chg); Py_XDECREF(s->o_tim);
  Py_XDECREF(s->o_pos); Py_XDECREF(s->o_mom);
  Py_TYPE(s)->tp_free((PyObject *)s);
}
static PyTypeObject InputEventsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "straw_detector.InputEvents",
    .tp_basicsize = sizeof(InputEventsObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)InputEvents_init,
    .tp_dealloc = (destructor)InputEvents_dealloc,
};

/* ---- DebugBuffers: optional batched debug outputs (any field None to skip) ---- */
typedef struct {
  PyObject_HEAD
  PyObject *o_traj, *o_proc, *o_ti, *o_tf, *o_te, *o_tc;  // INCREF'd
  npy_float *traj; npy_intp trs0, trs1, trs2, trs3; int n_traj, max_particles;
  int *proc_ids;
  int *tree_int; npy_float *tree_float; int *tree_event; int *tree_count; int tree_cap;
} DebugBuffersObject;

static int DebugBuffers_init(DebugBuffersObject *s, PyObject *args, PyObject *kwds) {
  (void)kwds;
  PyObject *tj, *pi, *ti, *tf, *te, *tc;
  if (!PyArg_ParseTuple(args, "OOOOOO", &tj, &pi, &ti, &tf, &te, &tc)) return -1;
  if (!Py_IsNone(tj)) {
    PyArrayObject *a = (PyArrayObject *)tj;
    if (!PyArray_Check(tj) || PyArray_TYPE(a) != NPY_FLOAT32 || PyArray_NDIM(a) != 4 ||
        PyArray_DIM(a, 3) != SPACE_DIM) {
      PyErr_SetString(PyExc_TypeError, "trajectories must be (n, max_particles, n_traj, 3) float32");
      return -1;
    }
    s->traj = (npy_float *)PyArray_DATA(a);
    s->trs0 = PyArray_STRIDE(a, 0) / sizeof(npy_float);
    s->trs1 = PyArray_STRIDE(a, 1) / sizeof(npy_float);
    s->trs2 = PyArray_STRIDE(a, 2) / sizeof(npy_float);
    s->trs3 = PyArray_STRIDE(a, 3) / sizeof(npy_float);
    s->max_particles = (int)PyArray_DIM(a, 1);
    s->n_traj = (int)PyArray_DIM(a, 2);
    Py_INCREF(tj); s->o_traj = tj;
  }
  if (!Py_IsNone(pi)) {
    PyArrayObject *a = (PyArrayObject *)pi;
    if (!PyArray_Check(pi) || PyArray_TYPE(a) != NPY_INT32 || PyArray_NDIM(a) != 2) {
      PyErr_SetString(PyExc_TypeError, "process_ids must be a (n, M) int32 array");
      return -1;
    }
    s->proc_ids = (int *)PyArray_DATA(a);
    Py_INCREF(pi); s->o_proc = pi;
  }
  if (!Py_IsNone(ti)) {
    PyArrayObject *ai = (PyArrayObject *)ti, *af = (PyArrayObject *)tf, *ae = (PyArrayObject *)te,
                  *ac = (PyArrayObject *)tc;
    if (!PyArray_Check(ti) || PyArray_TYPE(ai) != NPY_INT32 || PyArray_NDIM(ai) != 2 ||
        PyArray_DIM(ai, 1) != 4 || !PyArray_Check(tf) || PyArray_TYPE(af) != NPY_FLOAT32 ||
        PyArray_NDIM(af) != 2 || PyArray_DIM(af, 1) != 7 || PyArray_DIM(af, 0) != PyArray_DIM(ai, 0) ||
        !PyArray_Check(te) || PyArray_TYPE(ae) != NPY_INT32 || PyArray_NDIM(ae) != 1 ||
        PyArray_DIM(ae, 0) != PyArray_DIM(ai, 0) || !PyArray_Check(tc) ||
        PyArray_TYPE(ac) != NPY_INT32 || PyArray_SIZE(ac) != 1) {
      PyErr_SetString(PyExc_TypeError, "tree_int (cap,4) i32, tree_float (cap,7) f32, tree_event (cap,) i32, tree_count (1,) i32");
      return -1;
    }
    s->tree_int = (int *)PyArray_DATA(ai);
    s->tree_float = (npy_float *)PyArray_DATA(af);
    s->tree_event = (int *)PyArray_DATA(ae);
    s->tree_count = (int *)PyArray_DATA(ac);
    s->tree_cap = (int)PyArray_DIM(ai, 0);
    Py_INCREF(ti); s->o_ti = ti;
    Py_INCREF(tf); s->o_tf = tf;
    Py_INCREF(te); s->o_te = te;
    Py_INCREF(tc); s->o_tc = tc;
  }
  return 0;
}
static void DebugBuffers_dealloc(DebugBuffersObject *s) {
  Py_XDECREF(s->o_traj); Py_XDECREF(s->o_proc);
  Py_XDECREF(s->o_ti); Py_XDECREF(s->o_tf); Py_XDECREF(s->o_te); Py_XDECREF(s->o_tc);
  Py_TYPE(s)->tp_free((PyObject *)s);
}
static PyTypeObject DebugBuffersType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "straw_detector.DebugBuffers",
    .tp_basicsize = sizeof(DebugBuffersObject),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)DebugBuffers_init,
    .tp_dealloc = (destructor)DebugBuffers_dealloc,
};

// see documentation for the python method
static PyObject *solve(PyObject *self, PyObject *args) {
  // Init-once objects: SimParams/Layout/InputEvents/Scratch + optional DebugBuffers.
  PyObject *py_sim_params = NULL, *py_layout = NULL, *py_input_events = NULL, *py_scratch = NULL, *py_debug = NULL;
  // Explicit per-call arrays: seeds, (n,2) boundaries, the per-layer design (z + angle
  // + peak B), and the non-debug outputs X/mask.
  PyObject *py_seeds = NULL, *py_boundaries = NULL, *py_layers = NULL, *py_angles = NULL, *py_B = NULL;
  PyObject *py_X = NULL, *py_mask = NULL;

  if (!PyArg_UnpackTuple(
          args, "straw_solve", 12, 12, &py_sim_params, &py_layout, &py_input_events, &py_scratch,
          &py_seeds, &py_boundaries, &py_layers, &py_angles, &py_B, &py_X, &py_mask, &py_debug)) {
    return NULL;
  }
  if (!PyObject_TypeCheck(py_sim_params, &SimParamsType) || !PyObject_TypeCheck(py_layout, &LayoutType) ||
      !PyObject_TypeCheck(py_input_events, &InputEventsType) || !PyObject_TypeCheck(py_scratch, &ScratchType)) {
    PyErr_SetString(PyExc_TypeError, "expected (SimParams, Layout, InputEvents, Scratch) for the first four args");
    return NULL;
  }
  if (!Py_IsNone(py_debug) && !PyObject_TypeCheck(py_debug, &DebugBuffersType)) {
    PyErr_SetString(PyExc_TypeError, "debug must be a DebugBuffers or None");
    return NULL;
  }
  const SimParamsObject *sp = (const SimParamsObject *)py_sim_params;
  const LayoutObject *ly = (const LayoutObject *)py_layout;
  const InputEventsObject *ie = (const InputEventsObject *)py_input_events;
  const ScratchObject *scr = (const ScratchObject *)py_scratch;
  const DebugBuffersObject *dbg = Py_IsNone(py_debug) ? NULL : (const DebugBuffersObject *)py_debug;

  SimConfig cfg = sp->cfg;  // physics + solver constants (one struct copy); Layout/debug fields overlaid below
  HitBuffers out = {0};     // mutable output + per-straw scratch
  cfg.n_views = ly->n_views; cfg.n_lpv = ly->n_lpv; cfg.layer_y_offset = ly->layer_y_offset;
  const npy_intp n_layers = ly->n_layers, n_straws = ly->n_straws;

  // Event boundaries (n,2) int32: event l = pool rows [boundaries[2l], boundaries[2l+1]).
  if (!PyArray_Check(py_boundaries) || PyArray_TYPE((PyArrayObject *)py_boundaries) != NPY_INT32 ||
      PyArray_NDIM((PyArrayObject *)py_boundaries) != 2 || PyArray_DIM((PyArrayObject *)py_boundaries, 1) != 2) {
    PyErr_SetString(PyExc_TypeError, "boundaries must be a (n, 2) int32 array");
    return NULL;
  }
  const npy_intp n_batch = PyArray_DIM((PyArrayObject *)py_boundaries, 0);
  const npy_int32 *boundaries = (const npy_int32 *)PyArray_DATA((PyArrayObject *)py_boundaries);

  // Per-event RNG seeds (n,) uint32 -- each event seeded independently (parallel-safe).
  if (!PyArray_Check(py_seeds) || PyArray_TYPE((PyArrayObject *)py_seeds) != NPY_UINT32 ||
      PyArray_NDIM((PyArrayObject *)py_seeds) != 1 || PyArray_DIM((PyArrayObject *)py_seeds, 0) != n_batch) {
    PyErr_SetString(PyExc_TypeError, "seeds must be a (n,) uint32 array");
    return NULL;
  }
  const npy_uint32 *seeds = (const npy_uint32 *)PyArray_DATA((PyArrayObject *)py_seeds);

  // Non-debug dense outputs: X (n, M, 5), mask (n, M). M = X.shape[1]. (No counts --
  // the per-event hit count is mask[e].sum().)
  if (!PyArray_Check(py_X) || PyArray_NDIM((PyArrayObject *)py_X) != 3 ||
      PyArray_DIM((PyArrayObject *)py_X, 0) != n_batch || PyArray_DIM((PyArrayObject *)py_X, 2) != 5) {
    PyErr_SetString(PyExc_TypeError, "X must be a (n, M, 5) float32 array");
    return NULL;
  }
  out.X = (npy_float *)PyArray_DATA((PyArrayObject *)py_X);
  out.mask = (int *)PyArray_DATA((PyArrayObject *)py_mask);
  cfg.M = (int)PyArray_DIM((PyArrayObject *)py_X, 1);

  // Optional debug outputs from the DebugBuffers object (dbg == NULL -> none). The
  // buffers were validated in DebugBuffers.__init__; here we just cross-check n/M and
  // wire them in. max_particles only gates the (debug) trajectory write -- it is large
  // when no trajectory buffer is present, so it never affects X/mask.
  int tree_overflow = 0;
  npy_float *trajectories = NULL;
  npy_intp trs0 = 0, trs1 = 0, trs2 = 0, trs3 = 0;
  cfg.max_particles = 1 << 30;
  cfg.n_traj = 0;
  cfg.step_t = 0.0f;
  if (dbg != NULL) {
    if (dbg->o_proc != NULL) {
      if (PyArray_DIM((PyArrayObject *)dbg->o_proc, 0) != n_batch ||
          PyArray_DIM((PyArrayObject *)dbg->o_proc, 1) != cfg.M) {
        PyErr_SetString(PyExc_TypeError, "DebugBuffers.process_ids must be (n, M)");
        return NULL;
      }
      out.process_ids = dbg->proc_ids;
    }
    if (dbg->o_ti != NULL) {
      out.tree_int = dbg->tree_int; out.tree_float = dbg->tree_float;
      out.tree_event = dbg->tree_event; out.tree_count = dbg->tree_count;
      out.tree_cap = dbg->tree_cap; out.tree_overflow = &tree_overflow;
    }
    if (dbg->o_traj != NULL) {
      if (PyArray_DIM((PyArrayObject *)dbg->o_traj, 0) != n_batch) {
        PyErr_SetString(PyExc_TypeError, "DebugBuffers.trajectories must have a leading dim of n");
        return NULL;
      }
      trajectories = dbg->traj;
      trs0 = dbg->trs0; trs1 = dbg->trs1; trs2 = dbg->trs2; trs3 = dbg->trs3;
      cfg.max_particles = dbg->max_particles;
      cfg.n_traj = dbg->n_traj;
      cfg.step_t = (cfg.n_traj > 0) ? cfg.max_time / (float)cfg.n_traj : 0.0f;
    }
  }

  // Particle pool: validated + ref-held by InputEvents; alias its pointers/strides.
  // Boundaries index rows in [0, T).
  const npy_float *initial_positions = ie->pos, *initial_momenta = ie->mom;
  const npy_float *masses = ie->mass, *charges = ie->chg, *initial_times = ie->tim;
  const npy_intp T = ie->T;
  const npy_intp ips0 = ie->ips0, ips1 = ie->ips1, ivs0 = ie->ivs0, ivs1 = ie->ivs1;
  const npy_intp ms0 = ie->ms0, chs0 = ie->chs0, its0 = ie->its0;

  // Per-event design (varies per proposal): layers/angles (n, n_layers) + peak B (n,).
  // z0/B_sigma/widths/heights are fixed (in Layout), so only these are passed.
  const PyArrayObject *layers_array = check_scalar_array(py_layers, n_batch, n_layers);
  const PyArrayObject *angles_array = check_scalar_array(py_angles, n_batch, n_layers);
  const PyArrayObject *B_array = check_design_array(py_B, n_batch);
  if (layers_array == NULL || angles_array == NULL || B_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "design: layers/angles must be (n, n_layers) float32, B (n,) float32");
    return NULL;
  }
  // dt (fixed step or 0 -> adaptive) came from SimParams (cfg.dt_fixed already set).
  const npy_float *Bs = PyArray_DATA(B_array);
  const npy_float *layers = PyArray_DATA(layers_array);
  const npy_float *angles = PyArray_DATA(angles_array);
  npy_intp Bs0 = PyArray_STRIDE(B_array, 0) / sizeof(npy_float);
  npy_intp ls0 = PyArray_STRIDE(layers_array, 0) / sizeof(npy_float);
  npy_intp ls1 = PyArray_STRIDE(layers_array, 1) / sizeof(npy_float);
  npy_intp as0 = PyArray_STRIDE(angles_array, 0) / sizeof(npy_float);
  npy_intp as1 = PyArray_STRIDE(angles_array, 1) / sizeof(npy_float);

  // trajectories + its strides (trs*) were set from the DebugBuffers object above.
  out.trajectories = trajectories;
  out.trs0 = trs0;
  out.trs1 = trs1;
  out.trs2 = trs2;
  out.trs3 = trs3;

  // Per-straw min-TDC scratch from the Scratch object (allocated once in Python, reused;
  // tdc pre-filled with the empty sentinel and kept clean by emit_event's sparse-clear).
  const size_t n_cells = (size_t)n_layers * (size_t)n_straws;
  if ((size_t)scr->n_cells != n_cells) {
    PyErr_SetString(PyExc_TypeError, "Scratch size must equal n_layers * n_straws");
    return NULL;
  }
  out.n_straws = (int)n_straws;
  out.n_fired = 0;
  out.tdc = scr->tdc;
  out.proc = scr->proc;
  out.fired_idx = scr->fired_idx;
  // Over-capacity partial-sort scratch: one malloc per call (not per event).
  TdcIdx *sel = (TdcIdx *)malloc(n_cells * sizeof(TdcIdx));
  if (sel == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS

  // printf("\n=== Starting solve: n_batch=%ld, n_particles=%ld, n_layers=%ld, n_straws=%ld ===\n",
         // n_batch, n_particles, n_layers, n_straws);

  for (int l = 0; l < n_batch; ++l) {
    const npy_float B = Bs[l * Bs0];
    // Event-local RNG key (independent per event -> reentrant / parallelisable); split
    // to seed each primary and the noise draw with their own independent streams.
    uint64_t event_key = seeds[l];

    // This event's geometry + field, shared by all its particles. z0/B_sigma + the
    // straw half-dimensions are fixed (from Layout); only B + layers/angles vary.
    const Geometry geom = {
        .layers = layers, .angles = angles,
        .ls0 = ls0, .ls1 = ls1, .as0 = as0, .as1 = as1,
        .n_layers = (int)n_layers, .n_straws = (int)n_straws,
        .z0_field = ly->z0, .B_sigma = ly->B_sigma, .B = B,
        .layer_width = ly->layer_width, .layer_height = ly->layer_height,
    };

    // printf("\nEvent %d: B=%.4f T\n", l, B);

    // This event's primaries are the pool rows [boundaries[l,0], boundaries[l,1]).
    // (uncapped: all the event's daughters; secondaries take slots [n_prim, ...)).
    const npy_intp p_start = boundaries[2 * l];
    const npy_intp p_end = boundaries[2 * l + 1];
    if (p_start < 0 || p_end > T || p_end < p_start) {
      // out-of-range boundary: skip this event (leave its output empty)
      emit_event(&cfg, &out, l, sel);
      continue;
    }
    const int n_primaries = (int)(p_end - p_start);

    int next_secondary_idx = n_primaries;  // Secondaries start after primaries

    for (int i = 0; i < n_primaries; ++i) {
      const npy_intp r = p_start + i;  // flat row for this primary

      npy_float x0 = initial_positions[r * ips0];
      npy_float y0 = initial_positions[r * ips0 + ips1];
      npy_float z0 = initial_positions[r * ips0 + 2 * ips1];

      npy_float px0 = initial_momenta[r * ivs0];
      npy_float py0 = initial_momenta[r * ivs0 + ivs1];
      npy_float pz0 = initial_momenta[r * ivs0 + 2 * ivs1];

      npy_float charge = charges[r * chs0];
      npy_float mass = masses[r * ms0];
      npy_float t_initial = initial_times[r * its0];

      // Track this primary particle (depth=0): parent -1, process kPPrimary.
      track_particle(
          &cfg, &geom, &out,
          l, i,
          -1, PROC_PRIMARY,
          x0, y0, z0,
          px0, py0, pz0,
          mass, charge,
          t_initial,
          0, 0,  // start_step=0, depth=0
          &next_secondary_idx, split_key(&event_key)
      );
    }

    // Uncorrelated detector noise: a few random straw hits per event, recorded into
    // the per-straw scratch like any other hit. Generic electronic noise -- NOT a
    // model of the full-sim hit_track==-2 hits (those are real shower secondaries).
    if (cfg.noise_rate > 0.0f) {
      uint64_t noise_rng = split_key(&event_key);
      const int n_noise = rand_poisson(&noise_rng, cfg.noise_rate);
      for (int q = 0; q < n_noise; ++q) {
        int k = (int)(rand01(&noise_rng) * n_layers);
        if (k >= n_layers) k = n_layers - 1;
        int straw_i = (int)(rand01(&noise_rng) * n_straws);
        if (straw_i >= n_straws) straw_i = n_straws - 1;
        const float t_noise = 278.0f + rand01(&noise_rng) * 160.0f; /* plausible TDC window (ns) */
        record_hit(&out, k, straw_i, t_noise, PROC_NULL);
      }
    }

    // Emit this event's fired straws (min-TDC dedup) -> dense X/mask, keeping the M
    // earliest; then sparse-clear the scratch for the next event.
    emit_event(&cfg, &out, l, sel);
  }

  Py_END_ALLOW_THREADS

  free(sel);  // tdc/proc/fired_idx are caller-owned (passed in), not freed here

  // Debug tree capacity exceeded: the flat buffers are too small for the particles
  // produced. The caller sizes them n_batch*max_particles, so this should not happen.
  if (tree_overflow) {
    PyErr_SetString(PyExc_ValueError,
                    "MC tree buffer overflow: more particles than tree capacity; enlarge the tree buffers.");
    return NULL;
  }

  /* Return 0 - dense X / mask were filled in-place */
  return PyLong_FromLong(0);
}

static PyMethodDef StrawDetectorMethods[] = {
    {"solve", solve, METH_VARARGS,
     "Solve equations of motion and computes detector's response."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef straw_detector_module = {
    PyModuleDef_HEAD_INIT, "straw_detector", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    StrawDetectorMethods};

PyMODINIT_FUNC PyInit_straw_detector(void) {
  PyObject *m;
  m = PyModule_Create(&straw_detector_module);
  if (m == NULL)
    return NULL;

  import_array();

  // Register the init-once solver objects.
  PyTypeObject *types[] = {&SimParamsType, &LayoutType, &InputEventsType, &DebugBuffersType, &ScratchType};
  const char *names[] = {"SimParams", "Layout", "InputEvents", "DebugBuffers", "Scratch"};
  for (int i = 0; i < 5; ++i) {
    if (PyType_Ready(types[i]) < 0) {
      Py_DECREF(m);
      return NULL;
    }
    Py_INCREF(types[i]);
    if (PyModule_AddObject(m, names[i], (PyObject *)types[i]) < 0) {
      Py_DECREF(types[i]);
      Py_DECREF(m);
      return NULL;
    }
  }

  return m;
}
