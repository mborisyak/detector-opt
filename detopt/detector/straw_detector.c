#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
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
    int event_idx,           // Event index
    int particle_idx,        // Particle index (for output identification)
    npy_float x0, npy_float y0, npy_float z0,           // Initial position (cm)
    npy_float px0, npy_float py0, npy_float pz0,        // Initial momentum (MeV/c)
    npy_float mass, npy_float charge,                    // Mass (MeV/c^2), charge (e)
    npy_float t_initial,     // Initial time of particle (ns)
    int start_step,          // Time step to start from
    int depth,               // Recursion depth (0 for primary)
    int max_depth,           // Max recursion depth
    // Detector geometry
    const npy_float *layers, const npy_float *heights,
    const npy_float *widths, const npy_float *angles,
    npy_float z0_field, npy_float B_sigma,
    int n_layers, int n_straws,
    npy_intp ls0, npy_intp ls1,
    npy_intp hs0, npy_intp hs1,
    npy_intp ws0, npy_intp ws1,
    npy_intp as0, npy_intp as1,
    // Simulation parameters
    npy_float dt, npy_float B, int n_steps,
    // Sparse output arrays
    int *sparse_events, int *sparse_particles, int *sparse_layers, int *sparse_straws,
    npy_float *sparse_values, npy_float *sparse_r_mm, npy_float *sparse_t0,
    npy_float *sparse_hit_pos, int *sparse_count,
    // Trajectories
    npy_float *trajectories,
    npy_intp trs0, npy_intp trs1, npy_intp trs2, npy_intp trs3,
    // Secondary production parameters
    float p_spawn_single, float p_spawn_pair, float E_sec_MeV,
    int *next_secondary_idx,  // Pointer to next available secondary index
    int max_particles
);

// Forward declaration for recursion
static void track_particle(
    int event_idx, int particle_idx,
    npy_float x0, npy_float y0, npy_float z0,
    npy_float px0, npy_float py0, npy_float pz0,
    npy_float mass, npy_float charge,
    npy_float t_initial,
    int start_step, int depth, int max_depth,
    const npy_float *layers, const npy_float *heights,
    const npy_float *widths, const npy_float *angles,
    npy_float z0_field, npy_float B_sigma,
    int n_layers, int n_straws,
    npy_intp ls0, npy_intp ls1, npy_intp hs0, npy_intp hs1,
    npy_intp ws0, npy_intp ws1, npy_intp as0, npy_intp as1,
    npy_float dt, npy_float B, int n_steps,
    int *sparse_events, int *sparse_particles, int *sparse_layers, int *sparse_straws,
    npy_float *sparse_values, npy_float *sparse_r_mm, npy_float *sparse_t0,
    npy_float *sparse_hit_pos, int *sparse_count,
    npy_float *trajectories,
    npy_intp trs0, npy_intp trs1, npy_intp trs2, npy_intp trs3,
    float p_spawn_single, float p_spawn_pair, float E_sec_MeV,
    int *next_secondary_idx, int max_particles
) {
  // Initialize RNG state per particle
  uint32_t rng_state = RNG_SEED_BASE + (uint32_t)(event_idx * 10000 + particle_idx + depth * 1000);

  npy_float x = x0;
  npy_float y = y0;
  npy_float z = z0;

  npy_float px = px0;
  npy_float py = py0;
  npy_float pz = pz0;

  // printf("track_particle: event=%d, particle=%d, depth=%d, pos=(%.2f,%.2f,%.2f), p=(%.2f,%.2f,%.2f), mass=%.2f, charge=%.2f\n",
  //        event_idx, particle_idx, depth, x, y, z, px, py, pz, mass, charge);

  // Check for ghost particles
  if (mass < SLOW || (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW)) {
    // printf("  -> GHOST PARTICLE, skipping\n");
    return;
  }

  // printf("  -> Valid particle, tracking through %d steps, start_step=%d, n_steps=%d\n", n_steps - start_step, start_step, n_steps);
  // printf("  -> z0_field=%.2f, B_sigma=%.2f, n_layers=%d, n_straws=%d\n", z0_field, B_sigma, n_layers, n_straws);

  npy_float p2 = px * px + py * py + pz * pz;
  const npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));
  // printf("  -> gamma=%.5f\n", gamma);
  npy_float vx = px / (gamma * mass); // v's are dimensionless, in units of c
  npy_float vy = py / (gamma * mass);
  npy_float vz = pz / (gamma * mass);
  //printf("  -> v=%.5f\n", sqrtf(vx * vx + vy * vy + vz * vz));

  const npy_float mass_MeV_kg = 1.78266192e-30f;
  const npy_float charge_e_C = 1.602176634e-19f;
  const npy_float c = 0.5 * (dt / 1e9) * (charge * charge_e_C) /
                      (mass * mass_MeV_kg) / gamma;

  for (int j = start_step; j < n_steps; ++j) {
    const npy_float Bx = B * exp(-square((z - z0_field) / B_sigma));
    const npy_float tx = c * Bx;
    const npy_float t_norm_sqr = tx * tx;

    // Boris pusher: rotated speed
    const npy_float vy_m = vy + vz * tx;
    const npy_float vz_m = vz - vy * tx;

    const npy_float sx = 2 * tx / (1 + t_norm_sqr);

    vy = vy_m + vz_m * sx;
    vz = vz_m - vy_m * sx;

    const npy_float dx = dt * vx * 29.9792f;
    const npy_float dy = dt * vy * 29.9792f;
    const npy_float dz = dt * vz * 29.9792f;

    const npy_float x_ = x + dx;
    const npy_float y_ = y + dy;
    const npy_float z_ = z + dz;
    // Debug
    //if (j % 50 == 0) printf("  Step %d: pos=(%.2f,%.2f,%.2f) -> (%.2f,%.2f,%.2f), time %f\n", j, x, y, z, x_, y_, z_, j * dt);


    for (int k = 0; k < n_layers; ++k) {
      const npy_float layer = layers[event_idx * ls0 + k * ls1];
      const npy_float height = heights[event_idx * hs0 + k * hs1];
      const npy_float width = widths[event_idx * ws0 + k * ws1];
      const npy_float r = height / n_straws;  // Straw radius (height is half-height)
      // Use larger acceptance window to prevent particles from stepping over layers
      // Particles moving at ~c with dt=0.1ns move ~3cm per step, so need margin > straw diameter
      const npy_float layer_half_thickness = 1.0f * r;  // 3x radius for safe margin
      const npy_float left = layer - layer_half_thickness;
      const npy_float right = layer + layer_half_thickness;

      // Debug output (disabled by default)
      // if (event_idx == 0  && k == 8) {
      //   printf("E%d P%d L%d Step%d: layer_z=%.2f, r=%.2f, bounds=[%.2f,%.2f], z=%.2f->%.2f\n",
      //          event_idx, particle_idx, k, j, layer, r, left, right, z, z_);
      // }

      // check for potential hit
      if ((z < left && z_ < left) || (z > right && z_ > right)) {
        // if (event_idx == 0 && particle_idx < 2 && k < 5 && j < 10) {
        //   printf("  -> SKIP: both before or both after layer\n");
        // }
        continue;
      }
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
      const npy_int straw_i_center = (npy_int)floor(0.5 * (ry + height) / r);

      // Particle segment: from (x,y,z) to (x_,y_,z_)
      const npy_float rx_ = nx * x_ + ny * y_;
      const npy_float ry_ = -ny * x_ + nx * y_;

      // Particle direction vector AB
      const npy_float drx = rx_ - rx;
      const npy_float dry = ry_ - ry;
      const npy_float dz = z_ - z;

      npy_int best_straw_i = -1;
      npy_float best_dist_sq = r * r + 1.0f;

      for (int straw_offset = -2; straw_offset <= 2; straw_offset++) {
        const npy_int straw_i = straw_i_center + straw_offset;
        if (!(straw_i >= 0 && straw_i < n_straws)) {
          continue;
        }

        const npy_float straw_y = (2 * straw_i + 1) * r - height;

        // Wire direction: along x-axis, CD = (1, 0, 0)
        // Vector from particle start to wire point: AC = (0, straw_y, layer) - (rx, ry, z)
        const npy_float acx = -rx;
        const npy_float acy = straw_y - ry;
        const npy_float acz = layer - z;

        // Cross product AB x CD where CD = (1, 0, 0)
        const npy_float cross_x = 0.0f;
        const npy_float cross_y = dz;
        const npy_float cross_z = -dry;
        const npy_float cross_norm = sqrtf(cross_y * cross_y + cross_z * cross_z);

        npy_float sqr_distance_to_wire;
        if (cross_norm < 1e-6f) {
          // Lines are parallel, use perpendicular distance
          sqr_distance_to_wire = acy * acy + acz * acz;
        } else {
          // Distance = |AC · (AB x CD)| / |AB x CD|
          const npy_float dot = acx * cross_x + acy * cross_y + acz * cross_z;
          const npy_float dist = fabsf(dot) / cross_norm;
          sqr_distance_to_wire = dist * dist;
        }

        // if (event_idx == 0 && particle_idx < 2 && k == 8) {
        //     printf(" DIST %f   r = %f straw=%d %f\n", sqr_distance_to_wire, r, straw_i, sqrtf(sqr_distance_to_wire));
        // }

        if (sqr_distance_to_wire < r * r && sqr_distance_to_wire < best_dist_sq) {
          best_straw_i = straw_i;
          best_dist_sq = sqr_distance_to_wire;
        }
      }

      if (best_straw_i < 0) {
        continue;
      }

      // if (event_idx == 0 && particle_idx < 2 && k == 8) {
      //     printf("  -> PASSED DIST %f\n", best_dist_sq);
      // }

      // Check if this straw was already hit by this particle
      int is_first_hit_in_straw = 1;
      for (int h = 0; h < sparse_count[0]; h++) {
        if (sparse_events[h] == event_idx && sparse_particles[h] == particle_idx &&
            sparse_layers[h] == k && sparse_straws[h] == best_straw_i) {
          is_first_hit_in_straw = 0;
          break;
        }
      }
      if (!is_first_hit_in_straw) {
        continue;
      }

      // if (event_idx == 0 && particle_idx < 2 && k == 8) {
      //     printf("  -> STORED\n");
      // }

      // Record hit to sparse arrays
      int sparse_idx = sparse_count[0];
      sparse_events[sparse_idx] = event_idx;
      sparse_particles[sparse_idx] = particle_idx;
      sparse_layers[sparse_idx] = k;
      sparse_straws[sparse_idx] = best_straw_i;
      sparse_values[sparse_idx] = dt;
      sparse_r_mm[sparse_idx] = sqrtf(best_dist_sq) * 10.0f;
      sparse_t0[sparse_idx] = t_initial + j * dt;

      sparse_hit_pos[sparse_idx * 3 + 0] = x;
      sparse_hit_pos[sparse_idx * 3 + 1] = y;
      sparse_hit_pos[sparse_idx * 3 + 2] = z;

      sparse_count[0]++;

      // if (particle_idx < 10) {
      //   printf("  HIT recorded: particle=%d, layer=%d, straw=%d, t0=%.2f ns, sparse_count=%d\n",
      //          particle_idx, k, straw_i, j * dt, sparse_count[0]);
      // }

      // Spawn secondaries on first hit in this straw (if not at max depth)
      if (is_first_hit_in_straw && depth < max_depth &&
          (p_spawn_single > 0.0f || p_spawn_pair > 0.0f)) {
        float r_spawn = rand01(&rng_state);

        // single electron first
        if (r_spawn < p_spawn_single) {
          // Get next secondary index - check bounds
          if (*next_secondary_idx >= max_particles) {
            // No more slots available for secondaries
            continue;
          }
          int e_idx = (*next_secondary_idx)++;

          // Random isotropic direction
          float u = rand01(&rng_state);
          float v = rand01(&rng_state);
          float cos_theta = 2.0f * u - 1.0f;
          float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
          float phi = 2.0f * (float)M_PI * v;

          const npy_float mass_e = 0.511f;
          const npy_float charge_e = -1.0f;
          npy_float p_sec = sqrtf(E_sec_MeV * E_sec_MeV + 2.0f * E_sec_MeV * mass_e);

          npy_float px_e = p_sec * sin_theta * cosf(phi);
          npy_float py_e = p_sec * sin_theta * sinf(phi);
          npy_float pz_e = p_sec * cos_theta;

          // Recursively track secondary from this position
          track_particle(
              event_idx, e_idx,
              x_, y_, z_,
              px_e, py_e, pz_e,
              mass_e, -1.0f,
              t_initial,
              j + 1, depth + 1, max_depth,
              layers, heights, widths, angles, z0_field, B_sigma,
              n_layers, n_straws,
              ls0, ls1, hs0, hs1, ws0, ws1, as0, as1,
              dt, B, n_steps,
              sparse_events, sparse_particles, sparse_layers, sparse_straws,
              sparse_values, sparse_r_mm, sparse_t0, sparse_hit_pos, sparse_count,
              trajectories, trs0, trs1, trs2, trs3,
              p_spawn_single, p_spawn_pair, E_sec_MeV,
              next_secondary_idx, max_particles
          );
        }
        // e+e- pair production
        else if (r_spawn < (p_spawn_single + p_spawn_pair)) {
          // Get indices for e- and e+ - check bounds
          if (*next_secondary_idx + 1 >= max_particles) {
            // Need 2 slots for pair, not enough available
            continue;
          }
          int e_minus_idx = (*next_secondary_idx)++;
          int e_plus_idx = (*next_secondary_idx)++;

          // Random direction for pair axis
          float u = rand01(&rng_state);
          float v = rand01(&rng_state);
          float cos_theta = 2.0f * u - 1.0f;
          float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
          float phi = 2.0f * (float)M_PI * v;

          const npy_float mass_e = 0.511f;
          const npy_float T_half = E_sec_MeV * 0.5f;
          npy_float p_sec = sqrtf(T_half * T_half + 2.0f * T_half * mass_e);

          // e- momentum
          npy_float px_em = p_sec * sin_theta * cosf(phi);
          npy_float py_em = p_sec * sin_theta * sinf(phi);
          npy_float pz_em = p_sec * cos_theta;

          // e+ momentum (opposite, back-to-back)
          npy_float px_ep = -px_em;
          npy_float py_ep = -py_em;
          npy_float pz_ep = -pz_em;

          // Recursively track e-
          track_particle(
            event_idx, e_minus_idx,
            x, y, z,
            px_em, py_em, pz_em,
            mass_e, -1.0f,
            t_initial,  // Inherit mother's initial time
            j, depth + 1, max_depth,
            layers, heights, widths, angles, z0_field, B_sigma,
            n_layers, n_straws,
            ls0, ls1, hs0, hs1, ws0, ws1, as0, as1,
            dt, B, n_steps,
            sparse_events, sparse_particles, sparse_layers, sparse_straws,
            sparse_values, sparse_r_mm, sparse_t0, sparse_hit_pos, sparse_count,
            trajectories, trs0, trs1, trs2, trs3,
            p_spawn_single, p_spawn_pair, E_sec_MeV,
            next_secondary_idx, max_particles
          );

          // Recursively track e+
          track_particle(
            event_idx, e_plus_idx,
            x, y, z,
            px_ep, py_ep, pz_ep,
            mass_e, +1.0f,
            t_initial,  // Inherit mother's initial time
            j, depth + 1, max_depth,
            layers, heights, widths, angles, z0_field, B_sigma,
            n_layers, n_straws,
            ls0, ls1, hs0, hs1, ws0, ws1, as0, as1,
            dt, B, n_steps,
            sparse_events, sparse_particles, sparse_layers, sparse_straws,
            sparse_values, sparse_r_mm, sparse_t0, sparse_hit_pos, sparse_count,
            trajectories, trs0, trs1, trs2, trs3,
            p_spawn_single, p_spawn_pair, E_sec_MeV,
            next_secondary_idx, max_particles
          );
        }
      }
    }

    x = x_;
    y = y_;
    z = z_;

    if (trajectories != NULL && particle_idx < max_particles) {  // Limit trajectory storage
      trajectories[event_idx * trs0 + particle_idx * trs1 + j * trs2] = x;
      trajectories[event_idx * trs0 + particle_idx * trs1 + j * trs2 + trs3] = y;
      trajectories[event_idx * trs0 + particle_idx * trs1 + j * trs2 + 2 * trs3] = z;
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
  PyObject *py_n_particles = NULL;
  PyObject *py_max_particles = NULL;
  PyObject *py_n_layers = NULL;
  PyObject *py_n_straws = NULL;

  PyObject *py_trajectories = NULL;

  // Sparse output arrays (pre-allocated from Python)
  PyObject *py_sparse_events = NULL;
  PyObject *py_sparse_particles = NULL;
  PyObject *py_sparse_layers = NULL;
  PyObject *py_sparse_straws = NULL;
  PyObject *py_sparse_values = NULL;
  PyObject *py_sparse_r_mm = NULL;
  PyObject *py_sparse_t0 = NULL;
  PyObject *py_sparse_hit_pos = NULL;
  PyObject *py_sparse_count = NULL;

  // Secondary particle parameters
  PyObject *py_p_spawn_single = NULL;
  PyObject *py_p_spawn_pair = NULL;
  PyObject *py_E_sec = NULL;

  if (!PyArg_UnpackTuple(
          args, "straw_solve", 32, 32, &py_initial_positions,
          &py_initial_momenta, &py_masses, &py_charges, &py_initial_times,
          &py_B, &py_z0,
          &py_B_sigma, &py_steps, &py_dt, &py_n_batch, &py_n_particles,
          &py_n_layers, &py_n_straws, &py_layers, &py_width, &py_heights,
          &py_angles, &py_trajectories, &py_sparse_events, &py_sparse_particles,
          &py_sparse_layers, &py_sparse_straws, &py_sparse_values,
          &py_sparse_r_mm, &py_sparse_t0, &py_sparse_hit_pos, &py_sparse_count,
          &py_p_spawn_single, &py_p_spawn_pair, &py_E_sec, &py_max_particles)) {
    return NULL;
  }

  // Get dimensions directly from parameters
  if (!PyLong_Check(py_steps)) {
    PyErr_SetString(PyExc_TypeError, "steps must be an int");
    return NULL;
  }
  const long n_steps = PyLong_AsLong(py_steps);

  if (!PyLong_Check(py_n_batch) || !PyLong_Check(py_n_particles) ||
      !PyLong_Check(py_n_layers) || !PyLong_Check(py_n_straws)) {
    PyErr_SetString(PyExc_TypeError, "n_batch, n_particles, n_layers, and n_straws must be ints");
    return NULL;
  }
  const npy_intp n_batch = PyLong_AsLong(py_n_batch);
  const npy_intp n_particles = PyLong_AsLong(py_n_particles);
  const npy_intp n_layers = PyLong_AsLong(py_n_layers);
  const npy_intp n_straws = PyLong_AsLong(py_n_straws);
  const npy_intp max_particles = PyLong_AsLong(py_max_particles);

  // Get sparse array pointers
  int *sparse_events = (int *)PyArray_DATA((PyArrayObject *)py_sparse_events);
  int *sparse_particles =
      (int *)PyArray_DATA((PyArrayObject *)py_sparse_particles);
  int *sparse_layers = (int *)PyArray_DATA((PyArrayObject *)py_sparse_layers);
  int *sparse_straws = (int *)PyArray_DATA((PyArrayObject *)py_sparse_straws);
  npy_float *sparse_values =
      (npy_float *)PyArray_DATA((PyArrayObject *)py_sparse_values);
  npy_float *sparse_r_mm =
      (npy_float *)PyArray_DATA((PyArrayObject *)py_sparse_r_mm);
  npy_float *sparse_t0 =
      (npy_float *)PyArray_DATA((PyArrayObject *)py_sparse_t0);
  npy_float *sparse_hit_pos =
      (npy_float *)PyArray_DATA((PyArrayObject *)py_sparse_hit_pos);
  int *sparse_count = (int *)PyArray_DATA((PyArrayObject *)py_sparse_count);

  /* Initialize sparse counter to 0 */
  sparse_count[0] = 0;

  // Get secondary parameters
  if (!PyFloat_Check(py_p_spawn_single) || !PyFloat_Check(py_p_spawn_pair) || !PyFloat_Check(py_E_sec)) {
    PyErr_SetString(PyExc_TypeError, "p_spawn_single, p_spawn_pair and E_sec must be floats");
    return NULL;
  }
  const float p_spawn_single = (float)PyFloat_AsDouble(py_p_spawn_single);
  const float p_spawn_pair = (float)PyFloat_AsDouble(py_p_spawn_pair);
  const float E_sec_MeV = (float)PyFloat_AsDouble(py_E_sec);

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
            PyArray_DIM(trajectories_array, 1) == n_particles &&
            PyArray_DIM(trajectories_array, 2) == n_steps &&
            PyArray_DIM(trajectories_array, 3) == SPACE_DIM)) {
      PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be a (n, n_particles, n_t, 3) float64 array.");
      return NULL;
    }
  } else {
    trajectories_array = NULL;
  }

  const PyArrayObject *initial_positions_array =
      check_vector_array(py_initial_positions, n_batch, n_particles);
  if (initial_positions_array == NULL) {
    PyErr_SetString(
        PyExc_TypeError,
        "initial_positions must be a (n, n_particles, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject *initial_momenta_array =
      check_vector_array(py_initial_momenta, n_batch, n_particles);
  if (initial_momenta_array == NULL) {
    PyErr_SetString(PyExc_TypeError,
                    "Invalid value for initial momenta provided. Must be a (n, "
                    "n_particles, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject *masses_array =
      check_scalar_array(py_masses, n_batch, n_particles);
  if (masses_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for masses provided. Must "
                                     "be a (n, n_particles) float64 array.");
    return NULL;
  }

  const PyArrayObject *charges_array =
      check_scalar_array(py_charges, n_batch, n_particles);
  if (charges_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for charges provided. Must "
                                     "be a (n, n_particles) float64 array.");
    return NULL;
  }

  const PyArrayObject *initial_times_array =
      check_scalar_array(py_initial_times, n_batch, n_particles);
  if (initial_times_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for initial_times provided. Must "
                                     "be a (n, n_particles) float64 array.");
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
  const npy_float dt = PyFloat_AsDouble(py_dt);

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

  npy_intp ips0 =
      PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  npy_intp ips1 =
      PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);
  npy_intp ips2 =
      PyArray_STRIDE(initial_positions_array, 2) / sizeof(npy_float);

  npy_intp ivs0 = PyArray_STRIDE(initial_momenta_array, 0) / sizeof(npy_float);
  npy_intp ivs1 = PyArray_STRIDE(initial_momenta_array, 1) / sizeof(npy_float);
  npy_intp ivs2 = PyArray_STRIDE(initial_momenta_array, 2) / sizeof(npy_float);

  npy_intp chs0 = PyArray_STRIDE(charges_array, 0) / sizeof(npy_float);
  npy_intp chs1 = PyArray_STRIDE(charges_array, 1) / sizeof(npy_float);

  npy_intp ms0 = PyArray_STRIDE(masses_array, 0) / sizeof(npy_float);
  npy_intp ms1 = PyArray_STRIDE(masses_array, 1) / sizeof(npy_float);

  npy_intp its0 = PyArray_STRIDE(initial_times_array, 0) / sizeof(npy_float);
  npy_intp its1 = PyArray_STRIDE(initial_times_array, 1) / sizeof(npy_float);

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

  Py_BEGIN_ALLOW_THREADS

  // printf("\n=== Starting solve: n_batch=%ld, n_particles=%ld, n_layers=%ld, n_straws=%ld ===\n",
         // n_batch, n_particles, n_layers, n_straws);

  for (int l = 0; l < n_batch; ++l) {
    const npy_float B = Bs[l * Bs0];

    // printf("\nEvent %d: B=%.4f T\n", l, B);

    // Count actual primary particles in this event (non-zero mass)
    int n_primaries = 0;
    for (int i = 0; i < n_particles; ++i) {
      npy_float mass = masses[l * ms0 + i * ms1];
      if (mass > SLOW) {
        n_primaries = i + 1;  // Track highest valid particle index + 1
      }
    }

    // Track to keep secondary particle indices
    int next_secondary_idx = n_primaries;  // Secondaries start after primaries

    for (int i = 0; i < n_particles; ++i) {
      // Extract scalar initial conditions for this particle
      npy_float x0 = initial_positions[l * ips0 + i * ips1];
      npy_float y0 = initial_positions[l * ips0 + i * ips1 + ips2];
      npy_float z0 = initial_positions[l * ips0 + i * ips1 + 2 * ips2];

      npy_float px0 = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py0 = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz0 = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];

      npy_float charge = charges[l * chs0 + i * chs1];
      npy_float mass = masses[l * ms0 + i * ms1];
      npy_float t_initial = initial_times[l * its0 + i * its1];

      npy_float z0_field = z0s[l];
      npy_float B_sigma = B_sigmas[l];

      // Track this primary particle (depth=0)
      track_particle(
          l, i,  // event_idx, particle_idx
          x0, y0, z0,
          px0, py0, pz0,
          mass, charge,
          t_initial,  // Initial time from data
          0, 0, 2,  // start_step=0, depth=0, max_depth=2
          layers, heights, widths, angles, z0_field, B_sigma,
          n_layers, n_straws,
          ls0, ls1, hs0, hs1, ws0, ws1, as0, as1,
          dt, B, n_steps,
          sparse_events, sparse_particles, sparse_layers, sparse_straws,
          sparse_values, sparse_r_mm, sparse_t0, sparse_hit_pos, sparse_count,
          trajectories, trs0, trs1, trs2, trs3,
          p_spawn_single, p_spawn_pair, E_sec_MeV,
          &next_secondary_idx, max_particles
      );
    }
  }

  Py_END_ALLOW_THREADS

  // printf("\n=== solve complete: total hits recorded = %d ===\n\n", sparse_count[0]);

  /* Return 0 - sparse arrays were filled in-place */
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
