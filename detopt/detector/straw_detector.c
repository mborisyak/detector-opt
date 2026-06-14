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

#define SEC_SPAWN_PROB 0.0f      /* probability of spawning secondary per hit */
#define SEC_E_MEV 0.01f          /* secondary kinetic energy in MeV */
#define SEC_MAX_DEPTH 1          /* max recursion depth (1 = no recursion) */
#define RNG_SEED_BASE 123456789u /* base RNG seed */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline npy_float square(npy_float x) { return x * x; }

inline npy_float f32_abs(npy_float x) { return x > 0.0 ? x : -x; }

inline npy_float f32_min(npy_float x, npy_float y) { return x > y ? y : x; }

inline npy_float f32_max(npy_float x, npy_float y) { return x > y ? x : y; }

inline npy_float imin(npy_int x, npy_int y) { return x > y ? y : x; }

inline npy_float imax(npy_int x, npy_int y) { return x > y ? x : y; }

/* Simple RNG for secondary generation */
static inline uint32_t xorshift32_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x ? x : 0xdeadbeefu;
  return *state;
}

static inline float rand01(uint32_t *state) {
  uint32_t r = xorshift32_next(state);
  return (float)(r >> 8) * (1.0f / 16777216.0f);
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
  const npy_float *layers, *heights, *widths, *angles;
  npy_intp ls0, ls1, hs0, hs1, ws0, ws1, as0, as1;
  int n_layers, n_straws;
  npy_float z0_field, B_sigma, B;
} Geometry;

/* Mutable dense output, written in place: padded (n, M, 5) hits + (n, M) mask
 * (counts[event] is the per-event write cursor) and the (n, max_particles,
 * n_traj, 3) trajectory viz buffer (NULL to skip). */
typedef struct {
  npy_float *X;
  int *mask;
  int *counts;
  npy_float *trajectories;
  npy_intp trs0, trs1, trs2, trs3;
  // Per-event O(1) dedup indicator: fired[global_layer*fired_stride + straw] != 0
  // iff that straw already recorded a hit this event. Scratch; sparse-cleared
  // after each event by walking its hits (no per-event memset).
  uint8_t *fired;
  int fired_stride;  // = n_straws
} HitBuffers;

/* Append one dense hit for an event (with per-(event,straw) dedup); returns 1 if
 * written/already-present (i.e. a real hit exists), 0 only if the buffer is full. */
static inline int dense_record(const SimConfig *cfg, HitBuffers *out, int event_idx, int k, int straw_i, float fdigi) {
  const int per_station = cfg->n_views * cfg->n_lpv;
  const int station = (per_station > 0) ? k / per_station : 0;
  const int rem = (per_station > 0) ? k % per_station : 0;
  const int view = (cfg->n_lpv > 0) ? rem / cfg->n_lpv : 0;
  const int lpv = (cfg->n_lpv > 0) ? rem % cfg->n_lpv : 0;
  // O(1) dedup: has this (global layer, straw) already fired in this event?
  const npy_intp fi = (npy_intp)k * out->fired_stride + straw_i;
  if (out->fired[fi]) return 0;  // already present -> not a new hit (skip re-record / re-spawn)

  const int count = out->counts[event_idx];
  if (count >= cfg->M) return 0;  // event buffer full
  out->fired[fi] = 1;  // set only on an actual record, so the clear can walk the hit list
  const npy_intp base = (npy_intp)event_idx * cfg->M * 5 + (npy_intp)count * 5;
  out->X[base + 0] = (npy_float)station;
  out->X[base + 1] = (npy_float)view;
  out->X[base + 2] = (npy_float)lpv;
  out->X[base + 3] = (npy_float)straw_i;
  out->X[base + 4] = fdigi;
  out->mask[(npy_intp)event_idx * cfg->M + count] = 1;
  out->counts[event_idx] = count + 1;
  return 1;
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

/* Standard normal via Box-Muller (one of the pair). */
static inline float rand_normal(uint32_t *state, float mean, float sigma) {
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
static inline void apply_scatter(uint32_t *state, npy_float xX0, npy_float charge,
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
static inline int rand_poisson(uint32_t *state, float lam) {
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

// Helper function: check if point (x, y) is inside a convex quadrilateral
// (parallelogram)
int point_in_parallelogram(npy_float x, npy_float y,
                           const npy_float corners[4][2]) {
  int i, j, sign = 0;
  for (i = 0; i < 4; ++i) {
    j = (i + 1) % 4;
    npy_float dx = corners[j][0] - corners[i][0];
    npy_float dy = corners[j][1] - corners[i][1];
    npy_float px = x - corners[i][0];
    npy_float py = y - corners[i][1];
    npy_float cross = dx * py - dy * px;
    if (cross == 0)
      continue;
    if (sign == 0)
      sign = (cross > 0) ? 1 : -1;
    else if ((cross > 0 && sign < 0) || (cross < 0 && sign > 0))
      return 0;
  }
  return 1;
}

// Track a single particle through the detector + recursive secondary tracking
static void track_particle(
    const SimConfig *cfg,    // immutable simulation settings
    const Geometry *geom,    // this event's detector geometry
    HitBuffers *out,         // dense hit + trajectory output (mutated in place)
    int event_idx, int particle_idx,
    npy_float x0, npy_float y0, npy_float z0,     // initial position (cm)
    npy_float px0, npy_float py0, npy_float pz0,  // initial momentum (MeV/c)
    npy_float mass, npy_float charge,             // mass (MeV/c^2), charge (e)
    npy_float t_initial,                          // particle start time (ns)
    int start_step, int depth,                    // loop start index, recursion depth
    int *next_secondary_idx,                      // next free secondary slot (shared)
    uint32_t *rng_seed                            // advancing RNG seed stream (owned by solve)
);

// Forward declaration for recursion
static void track_particle(
    const SimConfig *cfg, const Geometry *geom, HitBuffers *out,
    int event_idx, int particle_idx,
    npy_float x0, npy_float y0, npy_float z0,
    npy_float px0, npy_float py0, npy_float pz0,
    npy_float mass, npy_float charge,
    npy_float t_initial,
    int start_step, int depth,
    int *next_secondary_idx,
    uint32_t *rng_seed
) {
  // Unpack the structs into the locals the physics body uses by bare name.
  const npy_float *layers = geom->layers, *heights = geom->heights;
  const npy_float *widths = geom->widths, *angles = geom->angles;
  const npy_intp ls0 = geom->ls0, ls1 = geom->ls1, hs0 = geom->hs0, hs1 = geom->hs1;
  const npy_intp ws0 = geom->ws0, ws1 = geom->ws1, as0 = geom->as0, as1 = geom->as1;
  const int n_layers = geom->n_layers, n_straws = geom->n_straws;
  const npy_float z0_field = geom->z0_field, B_sigma = geom->B_sigma, B = geom->B;
  const npy_float dt_fixed = cfg->dt_fixed;  // fixed-step override (>0); 0 -> adaptive
  const int n_steps = cfg->n_steps, max_depth = cfg->max_depth, max_particles = cfg->max_particles;
  const float delta_const = cfg->delta_const, t_wall = cfg->t_wall, delta_Tcut = cfg->delta_Tcut;
  const float scatter_xX0 = cfg->scatter_xX0;
  const float max_dt = cfg->max_dt, max_time = cfg->max_time, step_t = cfg->step_t, lambda_conv_cm = cfg->lambda_conv_cm;
  const int n_traj = cfg->n_traj, enable_decay = cfg->enable_decay;
  npy_float *trajectories = out->trajectories;
  const npy_intp trs0 = out->trs0, trs1 = out->trs1, trs2 = out->trs2, trs3 = out->trs3;

  // Per-particle RNG seed: consume one from the advancing stream owned by solve()
  // and advance it by the golden-ratio odd stride so adjacent draws decorrelate.
  uint32_t rng_state = *rng_seed;
  *rng_seed += 0x9E3779B9u;

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
          if (*next_secondary_idx + 1 < max_particles) {
            const int e_minus = (*next_secondary_idx)++;
            const int e_plus = (*next_secondary_idx)++;
            const npy_float pe = 0.5f * p_mag; /* split photon energy */
            const npy_float charges_pair[2] = {-1.0f, +1.0f};
            const int idx_pair[2] = {e_minus, e_plus};
            for (int s = 0; s < 2; ++s) {
              track_particle(
                  cfg, geom, out,
                  event_idx, idx_pair[s], x, y, z, ux * pe, uy * pe, uz * pe,
                  MASS_E, charges_pair[s], t_ph, 0, depth + 1,
                  next_secondary_idx, rng_seed);
            }
          }
          return; /* photon consumed at conversion */
        }
        x += ux * dl; /* propagate straight to the next step */
        y += uy * dl;
        z += uz * dl;
        t_ph += max_dt;
      }
    }
    return; /* no conversion (or disabled): photon leaves no hits */
  }

  // printf("  -> Valid particle, tracking through %d steps, start_step=%d, n_steps=%d\n", n_steps - start_step, start_step, n_steps);
  // printf("  -> z0_field=%.2f, B_sigma=%.2f, n_layers=%d, n_straws=%d\n", z0_field, B_sigma, n_layers, n_straws);

  npy_float p2 = px * px + py * py + pz * pz;
  const npy_float p_mag = sqrtf(p2);  // |p| (MeV/c), conserved under static B
  const npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));
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
  const npy_float beta = sqrtf(vx * vx + vy * vy + vz * vz);
  const npy_float qabs = f32_abs(charge);
  const npy_float sag_num =
      (qabs > SLOW && beta > SLOW)
          ? 4.0f * STRAW_SIGMA_SPATIAL * mass * gamma / (beta * C_CM_PER_NS * K_BORIS * qabs)
          : 1e30f;  // straight / neutral track -> field imposes no ceiling

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
  for (int k = 0; k < n_layers; ++k) {
    const npy_float lz = layers[event_idx * ls0 + k * ls1];
    const npy_float lh = heights[event_idx * hs0 + k * hs1];
    const npy_float lr = lh / n_straws;  // straw radius (height is half-height)
    const int lpv = (cfg->n_lpv > 0) ? (k % per_station) % cfg->n_lpv : 0;
    pl_z[k] = lz;
    pl_height[k] = lh;
    pl_width[k] = widths[event_idx * ws0 + k * ws1];
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

    const npy_float Bx = B * exp(-square((z - z0_field) / B_sigma));

    // Adaptive step: 0.9x the sagitta ceiling, clamped to max_dt. A fixed dt
    // (dt_fixed > 0) overrides the adaptive choice.
    npy_float dt_step = max_dt;
    if (dt_fixed > 0.0f) {
      dt_step = dt_fixed;
    } else if (Bx > SLOW) {
      const npy_float dt_sag = 0.9f * sqrtf(sag_num / Bx);
      if (dt_sag < dt_step) dt_step = dt_sag;
    }

    const npy_float c = K_BORIS * dt_step * charge / (mass * gamma);
    const npy_float tx = c * Bx;
    const npy_float t_norm_sqr = tx * tx;

    // Boris pusher: rotated speed
    const npy_float vy_m = vy + vz * tx;
    const npy_float vz_m = vz - vy * tx;

    const npy_float sx = 2 * tx / (1 + t_norm_sqr);

    vy = vy_m + vz_m * sx;
    vz = vz_m - vy_m * sx;

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

      const npy_float nx = cos(angle);
      const npy_float ny = sin(angle);
      const npy_float rx = nx * x + ny * y;
      const npy_float ry = -ny * x + nx * y;

      // Compute parallelogram corners in local (rx, ry) frame
      npy_float corners[4][2] = {{-width, -height},
                                 {-width, height},
                                 {width, height},
                                 {width, -height}};

      // outside the parallelogram (frame)
      if (!point_in_parallelogram(rx, ry, corners)) {
        continue;
      }
      // if (event_idx == 0 && particle_idx < 2 && k == 8) {
      //     printf("  -> PASSED POINTS\n");
      // }
      // Particle segment: from (x,y,z) to (x_,y_,z_)
      const npy_float ry_ = -ny * x_ + nx * y_;

      // Particle direction vector AB
      const npy_float dry = ry_ - ry;
      const npy_float dz = z_ - z;

      // The segment crosses the plane spanning transverse [ry, ry_]. Convert that
      // span to a straw index RANGE (a track crossing at an angle passes through
      // several adjacent tubes); the +/-1 pad covers the straw-radius overlap at
      // the ends. EVERY tube the segment actually enters (dist < r) fires -- not
      // just the single closest -- each with its own drift distance / TDC time.
      const npy_float ry_min = (ry < ry_) ? ry : ry_;
      const npy_float ry_max = (ry > ry_) ? ry : ry_;
      // Invert straw_y = (2i+1)*r - height + yoff -> i = floor((y + height - yoff)/(2r)).
      npy_int i_lo = (npy_int)floor(0.5 * (ry_min + height - yoff) / r) - 1;
      npy_int i_hi = (npy_int)floor(0.5 * (ry_max + height - yoff) / r) + 1;
      if (i_lo < 0) i_lo = 0;
      if (i_hi >= n_straws) i_hi = n_straws - 1;

      // Wire direction is along x (CD = (1,0,0)); AB x CD is invariant across the
      // straws in this layer, so hoist it out of the index loop.
      const npy_float cross_y = dz;
      const npy_float cross_z = -dry;
      const npy_float cross_norm = sqrtf(cross_y * cross_y + cross_z * cross_z);

      int fired_any = 0;
      for (npy_int straw_i = i_lo; straw_i <= i_hi; ++straw_i) {
        const npy_float straw_y = (2 * straw_i + 1) * r - height + yoff;

        // Vector from particle start to wire point: AC = (0, straw_y, layer) - (rx, ry, z).
        // The rx (=acx) component drops out below since CD = (1,0,0) -> cross_x = 0.
        const npy_float acy = straw_y - ry;
        const npy_float acz = layer - z;

        npy_float sqr_distance_to_wire;
        if (cross_norm < 1e-6f) {
          // Lines are parallel, use perpendicular distance
          sqr_distance_to_wire = acy * acy + acz * acz;
        } else {
          // Distance = |AC . (AB x CD)| / |AB x CD| (CD = (1,0,0) -> cross_x = 0)
          const npy_float dot = acy * cross_y + acz * cross_z;
          const npy_float dist = fabsf(dot) / cross_norm;
          sqr_distance_to_wire = dist * dist;
        }

        if (sqr_distance_to_wire >= r * r) continue;  // tube not entered -> no hit

        // Record the hit straight into the dense buffer (per-(event,straw) dedup).
        // FairShip TDC: t_MC + |Gaus(dist,sigma)|/v_drift + (wire_end - x)/c.
        const float dist_cm = sqrtf(sqr_distance_to_wire);
        const float t_drift = fabsf(rand_normal(&rng_state, dist_cm, STRAW_SIGMA_SPATIAL)) / STRAW_VDRIFT;
        const float t_prop = (width - x) / C_CM_PER_NS; /* width = widths[event,k] = +x readout end */
        const float fdigi = t_now + t_drift + t_prop;
        if (dense_record(cfg, out, event_idx, k, straw_i, fdigi)) fired_any = 1;
      }

      // Highland multiple scattering: one kick per layer crossing the track passes
      // through (it is inside the frame here). Deflects v for the onward steps;
      // off when scatter_xX0 <= 0.
      apply_scatter(&rng_state, scatter_xX0, charge, p_mag, beta, &vx, &vy, &vz);

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
          if (r_spawn < p_single && *next_secondary_idx < max_particles) {
            // delta-ray: energy ~ 1/T^2 on [Tcut, Tmax] (inverse-CDF), isotropic.
            const int e_idx = (*next_secondary_idx)++;
            const float uu = rand01(&rng_state);
            const float T = 1.0f / (1.0f / delta_Tcut - uu * (1.0f / delta_Tcut - 1.0f / Tmax));
            const float u = rand01(&rng_state), v = rand01(&rng_state);
            const float cos_theta = 2.0f * u - 1.0f;
            const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
            const float phi = 2.0f * (float)M_PI * v;
            const npy_float p_sec = sqrtf(T * T + 2.0f * T * MASS_E);
            track_particle(cfg, geom, out, event_idx, e_idx, x_, y_, z_,
                           p_sec * sin_theta * cosf(phi), p_sec * sin_theta * sinf(phi),
                           p_sec * cos_theta, MASS_E, -1.0f, t_initial, 0, depth + 1,
                           next_secondary_idx, rng_seed);
          } else if (r_spawn < p_single + p_pair && *next_secondary_idx + 1 < max_particles) {
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
            track_particle(cfg, geom, out, event_idx, e_minus_idx, x, y, z,
                           pxm, pym, pzm, MASS_E, -1.0f, t_initial, 0, depth + 1, next_secondary_idx, rng_seed);
            track_particle(cfg, geom, out, event_idx, e_plus_idx, x, y, z,
                           -pxm, -pym, -pzm, MASS_E, +1.0f, t_initial, 0, depth + 1, next_secondary_idx, rng_seed);
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
        if (rand01(&rng_state) < pdecay && *next_secondary_idx < max_particles) {
          const int mu_idx = (*next_secondary_idx)++;
          // Muon direction = current velocity direction; |p| conserved by B.
          const npy_float pmag = sqrtf(px * px + py * py + pz * pz);
          const npy_float vmag = sqrtf(vx * vx + vy * vy + vz * vz);
          const npy_float scale = (vmag > SLOW) ? pmag / vmag : 0.0f;
          track_particle(
              cfg, geom, out,
              event_idx, mu_idx, x, y, z, vx * scale, vy * scale, vz * scale,
              MASS_MU, charge, t_now, 0, depth + 1,
              next_secondary_idx, rng_seed);  // t_now = decay time; muon starts fresh
          return; /* parent stops at the decay point */
        }
      }
    }
  }
}

// see documentation for the python method
static PyObject *solve(PyObject *self, PyObject *args) {
  PyObject *py_dt = NULL;
  // field parameters
  PyObject *py_B = NULL;

  // particle parameters
  PyObject *py_initial_positions = NULL;
  PyObject *py_initial_momenta = NULL;
  PyObject *py_masses = NULL;
  PyObject *py_charges = NULL;
  PyObject *py_initial_times = NULL;

  PyObject *py_layers = NULL;
  PyObject *py_width = NULL;
  PyObject *py_heights = NULL;
  PyObject *py_angles = NULL;

  PyObject *py_z0 = NULL;
  PyObject *py_B_sigma = NULL;

  PyObject *py_steps = NULL;
  PyObject *py_n_batch = NULL;
  PyObject *py_offsets = NULL;  // (n_batch+1,) int32 CSR offsets into the flat particle pool
  PyObject *py_max_particles = NULL;
  PyObject *py_n_layers = NULL;
  PyObject *py_n_straws = NULL;

  PyObject *py_trajectories = NULL;

  // Secondary particle parameters
  PyObject *py_delta_const = NULL;
  PyObject *py_t_wall = NULL;
  PyObject *py_delta_Tcut = NULL;

  // Extra physics-process parameters (gamma conversion / decay / noise).
  PyObject *py_lambda_conv = NULL;
  PyObject *py_enable_decay = NULL;
  PyObject *py_noise_rate = NULL;

  // Dense output: the C code writes the padded (n_batch, M, 5) hit array + the
  // (n_batch, M) mask directly, using `counts` as the per-event write cursor.
  PyObject *py_X = NULL;
  PyObject *py_mask = NULL;
  PyObject *py_counts = NULL;
  PyObject *py_n_views = NULL;
  PyObject *py_n_lpv = NULL;
  PyObject *py_seed = NULL;  // caller-supplied RNG seed (propagated to all C randomness)
  PyObject *py_max_dt = NULL;   // upper clamp on the adaptive step (ns)
  PyObject *py_max_time = NULL;  // per-particle integration-time cap (ns)
  PyObject *py_scatter_xX0 = NULL;  // Highland material budget x/X0 per layer (<=0 off)
  PyObject *py_layer_y_offset = NULL;  // half-pitch stagger between layers (cm)

  if (!PyArg_UnpackTuple(
          args, "straw_solve", 36, 36, &py_initial_positions,
          &py_initial_momenta, &py_masses, &py_charges, &py_initial_times,
          &py_B, &py_z0,
          &py_B_sigma, &py_steps, &py_dt, &py_n_batch, &py_offsets,
          &py_n_layers, &py_n_straws, &py_layers, &py_width, &py_heights,
          &py_angles, &py_trajectories,
          &py_delta_const, &py_t_wall, &py_delta_Tcut, &py_max_particles,
          &py_lambda_conv, &py_enable_decay, &py_noise_rate,
          &py_X, &py_mask, &py_counts, &py_n_views, &py_n_lpv, &py_seed,
          &py_max_dt, &py_max_time, &py_scatter_xX0, &py_layer_y_offset)) {
    return NULL;
  }
  if (!PyFloat_Check(py_max_dt) || !PyFloat_Check(py_max_time) ||
      !PyFloat_Check(py_scatter_xX0) || !PyFloat_Check(py_layer_y_offset)) {
    PyErr_SetString(PyExc_TypeError, "max_dt, max_time, scatter_xX0 and layer_y_offset must be floats");
    return NULL;
  }
  SimConfig cfg = {0};   // immutable simulation settings, filled below
  HitBuffers out = {0};  // mutable dense + trajectory output
  cfg.max_dt = (float)PyFloat_AsDouble(py_max_dt);
  cfg.max_time = (float)PyFloat_AsDouble(py_max_time);
  cfg.scatter_xX0 = (float)PyFloat_AsDouble(py_scatter_xX0);
  cfg.layer_y_offset = (float)PyFloat_AsDouble(py_layer_y_offset);
  if (!PyLong_Check(py_seed)) {
    PyErr_SetString(PyExc_TypeError, "seed must be an int");
    return NULL;
  }
  // Running RNG seed stream, owned by solve() and advanced by every track_particle
  // entry + noise draw (passed by pointer, like next_secondary_idx).
  uint32_t rng_seed = (uint32_t)PyLong_AsUnsignedLongMask(py_seed);

  // Get dimensions directly from parameters
  if (!PyLong_Check(py_steps)) {
    PyErr_SetString(PyExc_TypeError, "steps must be an int");
    return NULL;
  }
  const long n_steps = PyLong_AsLong(py_steps);

  if (!PyLong_Check(py_n_batch) ||
      !PyLong_Check(py_n_layers) || !PyLong_Check(py_n_straws)) {
    PyErr_SetString(PyExc_TypeError, "n_batch, n_layers, and n_straws must be ints");
    return NULL;
  }
  const npy_intp n_batch = PyLong_AsLong(py_n_batch);
  const npy_intp n_layers = PyLong_AsLong(py_n_layers);
  const npy_intp n_straws = PyLong_AsLong(py_n_straws);
  const npy_intp max_particles = PyLong_AsLong(py_max_particles);

  // Sparse particle inputs: a flat pool indexed by event via CSR offsets.
  // offsets is (n_batch+1,) int32; offsets[n_batch] = total particle count T.
  if (!PyArray_Check(py_offsets) || PyArray_TYPE((PyArrayObject *)py_offsets) != NPY_INT32 ||
      PyArray_NDIM((PyArrayObject *)py_offsets) != 1 ||
      PyArray_DIM((PyArrayObject *)py_offsets, 0) != n_batch + 1) {
    PyErr_SetString(PyExc_TypeError, "offsets must be a (n_batch+1,) int32 array");
    return NULL;
  }
  const npy_int32 *offsets = (const npy_int32 *)PyArray_DATA((PyArrayObject *)py_offsets);
  const npy_intp n_total = (npy_intp)offsets[n_batch];

  // Secondary-production material constants (delta-ray rate prefactor, straw wall
  // thickness, delta-ray tracking cut).
  if (!PyFloat_Check(py_delta_const) || !PyFloat_Check(py_t_wall) || !PyFloat_Check(py_delta_Tcut)) {
    PyErr_SetString(PyExc_TypeError, "delta_const, t_wall and delta_Tcut must be floats");
    return NULL;
  }
  cfg.delta_const = (float)PyFloat_AsDouble(py_delta_const);
  cfg.t_wall = (float)PyFloat_AsDouble(py_t_wall);
  cfg.delta_Tcut = (float)PyFloat_AsDouble(py_delta_Tcut);

  // Extra physics-process config (set the per-call file-scope knobs).
  if (!PyFloat_Check(py_lambda_conv) || !PyFloat_Check(py_noise_rate)) {
    PyErr_SetString(PyExc_TypeError, "lambda_conv and noise_rate must be floats");
    return NULL;
  }
  if (!PyLong_Check(py_enable_decay)) {
    PyErr_SetString(PyExc_TypeError, "enable_decay must be an int");
    return NULL;
  }
  cfg.lambda_conv_cm = (float)PyFloat_AsDouble(py_lambda_conv);
  cfg.enable_decay = (int)PyLong_AsLong(py_enable_decay);
  cfg.noise_rate = (float)PyFloat_AsDouble(py_noise_rate);

  // Integration caps into the config.
  cfg.n_steps = (int)n_steps;
  cfg.max_particles = (int)max_particles;
  cfg.max_depth = 2;  // primary -> secondary -> tertiary

  // Dense output buffers + layer-index decomposition. The caller preallocates X
  // (zeroed), mask (zeroed) and counts (zeroed, the per-event write cursor).
  out.X = (npy_float *)PyArray_DATA((PyArrayObject *)py_X);
  out.mask = (int *)PyArray_DATA((PyArrayObject *)py_mask);
  out.counts = (int *)PyArray_DATA((PyArrayObject *)py_counts);
  cfg.M = (int)PyArray_DIM((PyArrayObject *)py_X, 1);
  cfg.n_views = (int)PyLong_AsLong(py_n_views);
  cfg.n_lpv = (int)PyLong_AsLong(py_n_lpv);

  PyArrayObject *trajectories_array;

  if (!Py_IsNone(py_trajectories)) {
    if (!PyArray_Check(py_trajectories)) {
      PyErr_SetString(PyExc_TypeError,
                      "The trajectories buffer must be an float64 array.");
      return NULL;
    }
    trajectories_array = (PyArrayObject *)py_trajectories;

    if (!(
            // PyArray_IS_C_CONTIGUOUS(trajectories_array) &&
            PyArray_TYPE(trajectories_array) == NPY_FLOAT32 &&
            PyArray_NDIM(trajectories_array) == 4 &&
            PyArray_DIM(trajectories_array, 0) == n_batch &&
            PyArray_DIM(trajectories_array, 1) == max_particles &&
            PyArray_DIM(trajectories_array, 2) >= 1 &&
            PyArray_DIM(trajectories_array, 3) == SPACE_DIM)) {
      PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be a (n, max_particles, n_t, 3) float32 array.");
      return NULL;
    }
  } else {
    trajectories_array = NULL;
  }

  // Trajectory is sampled evenly in time, decoupled from the physics step count
  // (n_steps is just the loop's safety cap). step_t = max_time / n_traj.
  cfg.n_traj = (trajectories_array != NULL) ? (int)PyArray_DIM(trajectories_array, 2) : 0;
  cfg.step_t = (cfg.n_traj > 0) ? cfg.max_time / (float)cfg.n_traj : 0.0f;

  // Sparse particle inputs: flat (T, ...) pools indexed by event via `offsets`.
  const PyArrayObject *initial_positions_array =
      check_flat_vector(py_initial_positions, n_total);
  if (initial_positions_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "initial_positions must be a (T, 3) float32 array.");
    return NULL;
  }

  const PyArrayObject *initial_momenta_array =
      check_flat_vector(py_initial_momenta, n_total);
  if (initial_momenta_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "initial_momenta must be a (T, 3) float32 array.");
    return NULL;
  }

  const PyArrayObject *masses_array = check_flat_scalar(py_masses, n_total);
  if (masses_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "masses must be a (T,) float32 array.");
    return NULL;
  }

  const PyArrayObject *charges_array = check_flat_scalar(py_charges, n_total);
  if (charges_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "charges must be a (T,) float32 array.");
    return NULL;
  }

  const PyArrayObject *initial_times_array = check_flat_scalar(py_initial_times, n_total);
  if (initial_times_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "initial_times must be a (T,) float32 array.");
    return NULL;
  }

  const PyArrayObject *layers_array =
      check_scalar_array(py_layers, n_batch, n_layers);
  if (layers_array == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Invalid value for layers' z-positions provided. Must be a "
                    "(n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject *width_array =
      check_scalar_array(py_width, n_batch, n_layers);
  if (width_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for widths provided. Must "
                                     "be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject *angles_array =
      check_scalar_array(py_angles, n_batch, n_layers);
  if (angles_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for angles provided. Must "
                                     "be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject *heights_array =
      check_scalar_array(py_heights, n_batch, n_layers);
  if (heights_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for heights provided. Must "
                                     "be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject *B_array = check_design_array(py_B, n_batch);
  if (B_array == NULL) {
    PyErr_SetString(
        PyExc_TypeError,
        "Invalid value for B provided. Must be a (n, ) float64 array.");
    return NULL;
  }
  const PyArrayObject *z0_array = check_design_array(py_z0, n_batch);
  if (z0_array == NULL) {
    PyErr_SetString(
        PyExc_TypeError,
        "Invalid value for z0 provided. Must be a (n, ) float64 array.");
    return NULL;
  }
  const PyArrayObject *B_sigma_array = check_design_array(py_B_sigma, n_batch);
  if (B_sigma_array == NULL) {
    PyErr_SetString(
        PyExc_TypeError,
        "Invalid value for B_sigma provided. Must be a (n, ) float64 array.");
    return NULL;
  }

  if (!PyFloat_Check(py_dt)) {
    PyErr_SetString(PyExc_TypeError, "dt must be a double");
    return NULL;
  }
  cfg.dt_fixed = (npy_float)PyFloat_AsDouble(py_dt);  // 0 -> adaptive per-step dt

  const npy_float *initial_positions = PyArray_DATA(initial_positions_array);
  const npy_float *initial_momenta = PyArray_DATA(initial_momenta_array);
  const npy_float *charges = PyArray_DATA(charges_array);
  const npy_float *masses = PyArray_DATA(masses_array);
  const npy_float *initial_times = PyArray_DATA(initial_times_array);

  const npy_float *Bs = PyArray_DATA(B_array);
  const npy_float *z0s = PyArray_DATA(z0_array);
  const npy_float *B_sigmas = PyArray_DATA(B_sigma_array);

  const npy_float *layers = PyArray_DATA(layers_array);
  const npy_float *widths = PyArray_DATA(width_array);
  const npy_float *angles = PyArray_DATA(angles_array);
  const npy_float *heights = PyArray_DATA(heights_array);

  npy_float *trajectories =
      (trajectories_array == NULL) ? NULL : PyArray_DATA(trajectories_array);

  npy_intp Bs0 = PyArray_STRIDE(B_array, 0) / sizeof(npy_float);

  // Flat pools: row stride (per particle) + element stride (per xyz component).
  npy_intp ips0 = PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  npy_intp ips1 = PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);

  npy_intp ivs0 = PyArray_STRIDE(initial_momenta_array, 0) / sizeof(npy_float);
  npy_intp ivs1 = PyArray_STRIDE(initial_momenta_array, 1) / sizeof(npy_float);

  npy_intp chs0 = PyArray_STRIDE(charges_array, 0) / sizeof(npy_float);
  npy_intp ms0 = PyArray_STRIDE(masses_array, 0) / sizeof(npy_float);
  npy_intp its0 = PyArray_STRIDE(initial_times_array, 0) / sizeof(npy_float);

  npy_intp ls0 = PyArray_STRIDE(layers_array, 0) / sizeof(npy_float);
  npy_intp ls1 = PyArray_STRIDE(layers_array, 1) / sizeof(npy_float);

  npy_intp hs0 = PyArray_STRIDE(heights_array, 0) / sizeof(npy_float);
  npy_intp hs1 = PyArray_STRIDE(heights_array, 1) / sizeof(npy_float);

  npy_intp ws0 = PyArray_STRIDE(width_array, 0) / sizeof(npy_float);
  npy_intp ws1 = PyArray_STRIDE(width_array, 1) / sizeof(npy_float);

  npy_intp as0 = PyArray_STRIDE(angles_array, 0) / sizeof(npy_float);
  npy_intp as1 = PyArray_STRIDE(angles_array, 1) / sizeof(npy_float);

  npy_intp trs0 = 0;
  npy_intp trs1 = 0;
  npy_intp trs2 = 0;
  npy_intp trs3 = 0;

  if (trajectories_array != NULL) {
    trs0 = PyArray_STRIDE(trajectories_array, 0) / sizeof(npy_float);
    trs1 = PyArray_STRIDE(trajectories_array, 1) / sizeof(npy_float);
    trs2 = PyArray_STRIDE(trajectories_array, 2) / sizeof(npy_float);
    trs3 = PyArray_STRIDE(trajectories_array, 3) / sizeof(npy_float);
  }

  out.trajectories = trajectories;
  out.trs0 = trs0;
  out.trs1 = trs1;
  out.trs2 = trs2;
  out.trs3 = trs3;

  // Per-event dedup indicator (global layer, straw). One constant per-call alloc;
  // calloc gives the initial all-zero state, then each event sparse-clears only the
  // bits it set, so it stays zero between events. Freed after the threaded region.
  out.fired_stride = (int)n_straws;
  out.fired = (uint8_t *)calloc((size_t)n_layers * (size_t)n_straws, sizeof(uint8_t));
  if (out.fired == NULL) {
    PyErr_NoMemory();
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS

  // printf("\n=== Starting solve: n_batch=%ld, n_particles=%ld, n_layers=%ld, n_straws=%ld ===\n",
         // n_batch, n_particles, n_layers, n_straws);

  for (int l = 0; l < n_batch; ++l) {
    const npy_float B = Bs[l * Bs0];

    // This event's geometry + field, shared by all its particles.
    const Geometry geom = {
        .layers = layers, .heights = heights, .widths = widths, .angles = angles,
        .ls0 = ls0, .ls1 = ls1, .hs0 = hs0, .hs1 = hs1,
        .ws0 = ws0, .ws1 = ws1, .as0 = as0, .as1 = as1,
        .n_layers = (int)n_layers, .n_straws = (int)n_straws,
        .z0_field = z0s[l], .B_sigma = B_sigmas[l], .B = B,
    };

    // printf("\nEvent %d: B=%.4f T\n", l, B);

    // This event's primaries are the flat rows [offsets[l], offsets[l+1]). The
    // loader caps each event at max_particles, so the local index i (used as the
    // trajectory slot) stays < max_particles; secondaries take slots [n_prim, ...).
    const npy_intp p_start = offsets[l];
    const npy_intp p_end = offsets[l + 1];
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

      // Track this primary particle (depth=0)
      track_particle(
          &cfg, &geom, &out,
          l, i,
          x0, y0, z0,
          px0, py0, pz0,
          mass, charge,
          t_initial,
          0, 0,  // start_step=0, depth=0
          &next_secondary_idx, &rng_seed
      );
    }

    // Uncorrelated detector noise: a few random straw hits per event, written
    // straight into the dense buffer. Generic electronic noise -- NOT a model of
    // the full-sim hit_track==-2 hits (those are real untracked shower secondaries).
    if (cfg.noise_rate > 0.0f) {
      uint32_t noise_rng = rng_seed;
      rng_seed += 0x9E3779B9u;
      const int n_noise = rand_poisson(&noise_rng, cfg.noise_rate);
      for (int q = 0; q < n_noise; ++q) {
        int k = (int)(rand01(&noise_rng) * n_layers);
        if (k >= n_layers) k = n_layers - 1;
        int straw_i = (int)(rand01(&noise_rng) * n_straws);
        if (straw_i >= n_straws) straw_i = n_straws - 1;
        const float t_noise = 278.0f + rand01(&noise_rng) * 160.0f; /* plausible TDC window (ns) */
        dense_record(&cfg, &out, l, k, straw_i, t_noise);
      }
    }

    // Restore the dedup indicator to all-zero by clearing only the bits this event
    // set (one per recorded hit) -- cheaper than a full memset per event.
    {
      const int cnt = out.counts[l];
      const npy_intp eb = (npy_intp)l * cfg.M * 5;
      const int per_station = cfg.n_views * cfg.n_lpv;
      for (int q = 0; q < cnt; ++q) {
        const npy_intp b = eb + (npy_intp)q * 5;
        const int kk = (int)out.X[b + 0] * per_station + (int)out.X[b + 1] * cfg.n_lpv + (int)out.X[b + 2];
        out.fired[(npy_intp)kk * out.fired_stride + (int)out.X[b + 3]] = 0;
      }
    }
  }

  Py_END_ALLOW_THREADS

  free(out.fired);

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

  return m;
}
