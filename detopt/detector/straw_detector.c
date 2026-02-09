#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define SPACE_DIM 3

#define SLOW_VZ 1.0e-3f
#define SLOW    1.0e-6f

/* secondary / spawning knobs */
#define SEC_SPAWN_PROB 0.0f//0.0005f       /* per-step spawn probability (demo value) */
#define SEC_E_MEV      0.01f      /* secondary kinetic energy in MeV (tiny)  */
#define SEC_MAX_DEPTH  1          /* depth of recursion for secondaries      */
#define RNG_SEED_BASE  123456789u /* base RNG seed                            */

/* -------------------------------------------------------------------------- */
/* small helpers                                                              */
/* -------------------------------------------------------------------------- */
static inline npy_float square(npy_float x) {
  return x * x;
}

static inline npy_float f32_abs(npy_float x) {
  return x > 0.0f ? x : -x;
}

/* -------------------------------------------------------------------------- */
/* array shape checkers                                                       */
/* -------------------------------------------------------------------------- */
static const PyArrayObject *
check_vector_array(const PyObject *object, int batch, int size) {
  if (!PyArray_Check(object)) return NULL;
  const PyArrayObject *array = (const PyArrayObject *)object;

  if (PyArray_TYPE(array) == NPY_FLOAT32 &&
      PyArray_NDIM(array) == 3 &&
      PyArray_DIM(array, 0) == batch &&
      PyArray_DIM(array, 1) == size &&
      PyArray_DIM(array, 2) == SPACE_DIM) {
    return array;
  }
  return NULL;
}

static const PyArrayObject *
check_scalar_array(const PyObject *object, int batch, int size) {
  if (!PyArray_Check(object)) return NULL;
  const PyArrayObject *array = (const PyArrayObject *)object;

  if (PyArray_TYPE(array) == NPY_FLOAT32 &&
      PyArray_NDIM(array) == 2 &&
      PyArray_DIM(array, 0) == batch &&
      PyArray_DIM(array, 1) == size) {
    return array;
  }
  return NULL;
}

static const PyArrayObject *
check_design_array(const PyObject *object, int batch) {
  if (!PyArray_Check(object)) return NULL;
  const PyArrayObject *array = (const PyArrayObject *)object;

  if (PyArray_TYPE(array) == NPY_FLOAT32 &&
      PyArray_NDIM(array) == 1 &&
      PyArray_DIM(array, 0) == batch) {
    return array;
  }
  return NULL;
}

/* -------------------------------------------------------------------------- */
/* geometry helper: is point in rotated parallelogram                         */
/* -------------------------------------------------------------------------- */
static int
point_in_parallelogram(npy_float x, npy_float y, const npy_float corners[4][2]) {
  int i, j, sign = 0;
  for (i = 0; i < 4; ++i) {
    j = (i + 1) % 4;
    npy_float dx = corners[j][0] - corners[i][0];
    npy_float dy = corners[j][1] - corners[i][1];
    npy_float px = x - corners[i][0];
    npy_float py = y - corners[i][1];
    npy_float cross = dx * py - dy * px;
    if (cross == 0.0f) continue;
    if (sign == 0) {
      sign = (cross > 0.0f) ? 1 : -1;
    } else if ((cross > 0.0f && sign < 0) || (cross < 0.0f && sign > 0)) {
      return 0;
    }
  }
  return 1;
}

/* -------------------------------------------------------------------------- */
/* sparse hit structure                                                       */
/* -------------------------------------------------------------------------- */
typedef struct {
  int event;
  int particle;
  int layer;
  int straw;
  npy_float value;
  npy_float edep;
  npy_float r_mm;
  npy_float t0;
  npy_float hit_pos[3];
} sparse_hit_t;

typedef struct {
  sparse_hit_t *hits;
  size_t capacity;
  size_t count;
} sparse_hit_buffer_t;

static sparse_hit_buffer_t *
sparse_buffer_create(size_t initial_capacity) {
  sparse_hit_buffer_t *buf = (sparse_hit_buffer_t *)malloc(sizeof(sparse_hit_buffer_t));
  if (!buf) return NULL;
  buf->capacity = initial_capacity > 0 ? initial_capacity : 1024;
  buf->count = 0;
  buf->hits = (sparse_hit_t *)malloc(sizeof(sparse_hit_t) * buf->capacity);
  if (!buf->hits) {
    free(buf);
    return NULL;
  }
  return buf;
}

static int
sparse_buffer_add(sparse_hit_buffer_t *buf, int event, int particle, int layer, int straw,
                  npy_float value, npy_float edep, npy_float r_mm, npy_float t0,
                  npy_float x, npy_float y, npy_float z) {
  if (!buf) return 0;

  /* check if hit already exists and accumulate */
  for (size_t i = 0; i < buf->count; ++i) {
    if (buf->hits[i].event == event &&
        buf->hits[i].particle == particle &&
        buf->hits[i].layer == layer &&
        buf->hits[i].straw == straw) {
      /* accumulate value and edep */
      buf->hits[i].value += value;
      buf->hits[i].edep += edep;
      /* keep first hit's r_mm, t0, position */
      return 1;
    }
  }

  /* new hit - check if we have space */
  if (buf->count >= buf->capacity) {
    /* buffer is full - cannot add more hits */
    return 0;
  }

  sparse_hit_t *hit = &buf->hits[buf->count++];
  hit->event = event;
  hit->particle = particle;
  hit->layer = layer;
  hit->straw = straw;
  hit->value = value;
  hit->edep = edep;
  hit->r_mm = r_mm;
  hit->t0 = t0;
  hit->hit_pos[0] = x;
  hit->hit_pos[1] = y;
  hit->hit_pos[2] = z;
  return 1;
}

static void
sparse_buffer_free(sparse_hit_buffer_t *buf) {
  if (buf) {
    if (buf->hits) free(buf->hits);
    free(buf);
  }
}

/* -------------------------------------------------------------------------- */
/* simple xorshift RNG                                                        */
/* -------------------------------------------------------------------------- */
static inline uint32_t
xorshift32_next(uint32_t *state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x ? x : 0xdeadbeefu;
  return *state;
}

static inline float
rand01(uint32_t *state) {
  uint32_t r = xorshift32_next(state);
  return (float)(r >> 8) * (1.0f / 16777216.0f);
}

/* -------------------------------------------------------------------------- */
/* forward declaration                                                        */
/* -------------------------------------------------------------------------- */
static void particle_pusher(
  int l, int i,
  npy_float *x_ptr, npy_float *y_ptr, npy_float *z_ptr,
  const npy_float vx,
  npy_float *vy_ptr, npy_float *vz_ptr,
  const npy_float px, const npy_float py, const npy_float pz,
  const npy_float mass, const npy_float charge, const npy_float gamma,
  const npy_float c, const npy_float dt, const int n_steps,
  const npy_float B, const npy_float z0, const npy_float B_sigma,
  const npy_float *layers, const npy_float *widths,
  const npy_float *angles, const npy_float *heights,
  const npy_intp ls0, const npy_intp ls1,
  const npy_intp hs0, const npy_intp hs1,
  const npy_intp ws0, const npy_intp ws1,
  const npy_intp as0, const npy_intp as1,
  const int n_layers, const int n_straws,
  npy_float *response, npy_float *trajectories,
  npy_float *edep, npy_float *r_mm, npy_float *t0, npy_float *hit_pos,
  const npy_intp rs0, const npy_intp rs1, const npy_intp rs2, const npy_intp rs3,
  const npy_intp trs0, const npy_intp trs1, const npy_intp trs2, const npy_intp trs3,
  int step_offset,
  npy_float *mask, const npy_intp mask_s0, const npy_intp mask_s1,
  uint32_t rng_state, int depth, int max_depth,
  float p_spawn, float E_sec_MeV,
  float p_pair,  /* probability of e+e- pair production */
  int *next_free_per_batch,
  int n_slots,
  /* sparse buffer (may be NULL, if provided, used instead of dense arrays) */
  sparse_hit_buffer_t *sparse_buf
);

/* -------------------------------------------------------------------------- */
/* helper: spawn e+e- pair                                                    */
/* -------------------------------------------------------------------------- */
static void spawn_pair(
  int l, int i,
  npy_float x, npy_float y, npy_float z,
  const npy_float px, const npy_float py, const npy_float pz,
  const npy_float mass, const npy_float charge, const npy_float gamma,
  const npy_float c, const npy_float dt, const int n_steps, const int j,
  const npy_float B, const npy_float z0, const npy_float B_sigma,
  const npy_float *layers, const npy_float *widths,
  const npy_float *angles, const npy_float *heights,
  const npy_intp ls0, const npy_intp ls1,
  const npy_intp hs0, const npy_intp hs1,
  const npy_intp ws0, const npy_intp ws1,
  const npy_intp as0, const npy_intp as1,
  const int n_layers, const int n_straws,
  npy_float *response, npy_float *trajectories,
  npy_float *edep, npy_float *r_mm, npy_float *t0, npy_float *hit_pos,
  const npy_intp rs0, const npy_intp rs1, const npy_intp rs2, const npy_intp rs3,
  const npy_intp trs0, const npy_intp trs1, const npy_intp trs2, const npy_intp trs3,
  int step_offset,
  npy_float *mask, const npy_intp mask_s0, const npy_intp mask_s1,
  uint32_t *rng_state, int depth, int max_depth,
  float p_spawn, float E_sec_MeV,
  float p_pair,
  int *next_free_per_batch, int n_slots,
  sparse_hit_buffer_t *sparse_buf
) {
  /* build child kinematics for e+e- pair */
  uint32_t child_state = *rng_state;

  /* isotropic direction for the pair axis */
  float u = rand01(&child_state);
  float v = rand01(&child_state);
  float cos_theta = 2.0f * u - 1.0f;
  float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
  float phi = 2.0f * (float)M_PI * v;
  float dir_x = sin_theta * cosf(phi);
  float dir_y = sin_theta * sinf(phi);
  float dir_z = cos_theta;

  /* electron/positron properties */
  const npy_float mass_e = 0.511f;
  const npy_float charge_e_minus = -1.0f;
  const npy_float charge_e_plus = 1.0f;

  /* Each particle gets half the energy (simplified) */
  const npy_float T = E_sec_MeV * 0.5f;
  npy_float p_sec = sqrtf(T * T + 2.0f * T * mass_e);

  /* e- momentum along direction */
  npy_float px_e_minus = p_sec * dir_x;
  npy_float py_e_minus = p_sec * dir_y;
  npy_float pz_e_minus = p_sec * dir_z;

  /* e+ momentum opposite (back-to-back) */
  npy_float px_e_plus = -px_e_minus;
  npy_float py_e_plus = -py_e_minus;
  npy_float pz_e_plus = -pz_e_minus;

  /* e- kinematics */
  npy_float gamma_e_minus = sqrtf(1.0f + (p_sec * p_sec) / (mass_e * mass_e));
  npy_float vx_e_minus = px_e_minus / (gamma_e_minus * mass_e);
  npy_float vy_e_minus = py_e_minus / (gamma_e_minus * mass_e);
  npy_float vz_e_minus = pz_e_minus / (gamma_e_minus * mass_e);

  /* e+ kinematics */
  npy_float gamma_e_plus = gamma_e_minus;  /* same momentum magnitude */
  npy_float vx_e_plus = px_e_plus / (gamma_e_plus * mass_e);
  npy_float vy_e_plus = py_e_plus / (gamma_e_plus * mass_e);
  npy_float vz_e_plus = pz_e_plus / (gamma_e_plus * mass_e);

  /* start both at parent position */
  npy_float x_child = x;
  npy_float y_child = y;
  npy_float z_child = z;

  /* Try to store e- */
  int can_store_e_minus = (next_free_per_batch != NULL) &&
                          (next_free_per_batch[l] < n_slots);

  if (can_store_e_minus) {
    int child_i_e_minus = next_free_per_batch[l];
    next_free_per_batch[l] += 1;

    particle_pusher(
      l, child_i_e_minus,
      &x_child, &y_child, &z_child,
      vx_e_minus,
      &vy_e_minus, &vz_e_minus,
      px_e_minus, py_e_minus, pz_e_minus,
      mass_e, charge_e_minus, gamma_e_minus,
      c, dt, n_steps - j,
      B, z0, B_sigma,
      layers, widths, angles, heights,
      ls0, ls1, hs0, hs1,
      ws0, ws1, as0, as1,
      n_layers, n_straws,
      response, trajectories,
      edep, r_mm, t0, hit_pos,
      rs0, rs1, rs2, rs3,
      trs0, trs1, trs2, trs3,
      step_offset + j,
      mask, mask_s0, mask_s1,
      child_state, depth + 1, max_depth,
      p_spawn, E_sec_MeV,
      p_pair,
      next_free_per_batch,
      n_slots,
      sparse_buf
    );
  } else {
    /* simulate but don't store */
    npy_float dummy_vy = vy_e_minus;
    npy_float dummy_vz = vz_e_minus;
    particle_pusher(
      l, i,
      &x_child, &y_child, &z_child,
      vx_e_minus,
      &dummy_vy, &dummy_vz,
      px_e_minus, py_e_minus, pz_e_minus,
      mass_e, charge_e_minus, gamma_e_minus,
      c, dt, n_steps - j,
      B, z0, B_sigma,
      layers, widths, angles, heights,
      ls0, ls1, hs0, hs1,
      ws0, ws1, as0, as1,
      n_layers, n_straws,
      NULL, NULL, NULL, NULL, NULL, NULL,  /* discard mode */
      0, 0, 0, 0,
      0, 0, 0, 0,
      step_offset + j,
      NULL, 0, 0,
      child_state, depth + 1, max_depth,
      p_spawn, E_sec_MeV,
      0.0f,  /* p_pair */
      NULL, 0,
      NULL
    );
  }

  /* Try to store e+ */
  uint32_t child_state_e_plus = child_state;  /* use different RNG state */
  int can_store_e_plus = (next_free_per_batch != NULL) &&
                         (next_free_per_batch[l] < n_slots);

  if (can_store_e_plus) {
    int child_i_e_plus = next_free_per_batch[l];
    next_free_per_batch[l] += 1;

    /* reset position for e+ */
    x_child = x;
    y_child = y;
    z_child = z;

    particle_pusher(
      l, child_i_e_plus,
      &x_child, &y_child, &z_child,
      vx_e_plus,
      &vy_e_plus, &vz_e_plus,
      px_e_plus, py_e_plus, pz_e_plus,
      mass_e, charge_e_plus, gamma_e_plus,
      c, dt, n_steps - j,
      B, z0, B_sigma,
      layers, widths, angles, heights,
      ls0, ls1, hs0, hs1,
      ws0, ws1, as0, as1,
      n_layers, n_straws,
      response, trajectories,
      edep, r_mm, t0, hit_pos,
      rs0, rs1, rs2, rs3,
      trs0, trs1, trs2, trs3,
      step_offset + j,
      mask, mask_s0, mask_s1,
      child_state_e_plus, depth + 1, max_depth,
      p_spawn, E_sec_MeV,
      p_pair,
      next_free_per_batch,
      n_slots,
      sparse_buf
    );
  } else {
    /* simulate but don't store */
    npy_float dummy_vy = vy_e_plus;
    npy_float dummy_vz = vz_e_plus;
    x_child = x;
    y_child = y;
    z_child = z;
    particle_pusher(
      l, i,
      &x_child, &y_child, &z_child,
      vx_e_plus,
      &dummy_vy, &dummy_vz,
      px_e_plus, py_e_plus, pz_e_plus,
      mass_e, charge_e_plus, gamma_e_plus,
      c, dt, n_steps - j,
      B, z0, B_sigma,
      layers, widths, angles, heights,
      ls0, ls1, hs0, hs1,
      ws0, ws1, as0, as1,
      n_layers, n_straws,
      NULL, NULL, NULL, NULL, NULL, NULL,  /* discard mode */
      0, 0, 0, 0,
      0, 0, 0, 0,
      step_offset + j,
      NULL, 0, 0,
      child_state_e_plus, depth + 1, max_depth,
      p_spawn, E_sec_MeV,
      0.0f,  /* p_pair */
      NULL, 0,
      NULL
    );
  }

  *rng_state = child_state_e_plus;
}

/* -------------------------------------------------------------------------- */
/* helper: spawn a secondary particle                                         */
/* -------------------------------------------------------------------------- */
static void spawn_secondary(
  int l, int i,
  npy_float x, npy_float y, npy_float z,
  const npy_float px, const npy_float py, const npy_float pz,
  const npy_float mass, const npy_float charge, const npy_float gamma,
  const npy_float c, const npy_float dt, const int n_steps, const int j,
  const npy_float B, const npy_float z0, const npy_float B_sigma,
  const npy_float *layers, const npy_float *widths,
  const npy_float *angles, const npy_float *heights,
  const npy_intp ls0, const npy_intp ls1,
  const npy_intp hs0, const npy_intp hs1,
  const npy_intp ws0, const npy_intp ws1,
  const npy_intp as0, const npy_intp as1,
  const int n_layers, const int n_straws,
  npy_float *response, npy_float *trajectories,
  npy_float *edep, npy_float *r_mm, npy_float *t0, npy_float *hit_pos,
  const npy_intp rs0, const npy_intp rs1, const npy_intp rs2, const npy_intp rs3,
  const npy_intp trs0, const npy_intp trs1, const npy_intp trs2, const npy_intp trs3,
  int step_offset,
  npy_float *mask, const npy_intp mask_s0, const npy_intp mask_s1,
  uint32_t *rng_state, int depth, int max_depth,
  float p_spawn, float E_sec_MeV,
  float p_pair,
  int *next_free_per_batch, int n_slots,
  sparse_hit_buffer_t *sparse_buf
) {
        /* build child kinematics */
  uint32_t child_state = *rng_state;

        /* isotropic direction */
        float u = rand01(&child_state);
        float v = rand01(&child_state);
        float cos_theta = 2.0f * u - 1.0f;
        float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
        float phi = 2.0f * (float)M_PI * v;
        float dir_x = sin_theta * cosf(phi);
        float dir_y = sin_theta * sinf(phi);
        float dir_z = cos_theta;

        /* electron-like secondary */
        const npy_float mass_e   = 0.511f;
        const npy_float charge_e = -1.0f;
        const npy_float T        = E_sec_MeV;
        npy_float p_sec = sqrtf(T * T + 2.0f * T * mass_e);

        npy_float px_sec = p_sec * dir_x;
        npy_float py_sec = p_sec * dir_y;
        npy_float pz_sec = p_sec * dir_z;

        npy_float gamma_sec = sqrtf(1.0f + (p_sec * p_sec) / (mass_e * mass_e));

        npy_float vx_sec = px_sec / (gamma_sec * mass_e);
        npy_float vy_sec = py_sec / (gamma_sec * mass_e);
        npy_float vz_sec = pz_sec / (gamma_sec * mass_e);

        /* start child at parent position */
        npy_float x_child = x;
        npy_float y_child = y;
        npy_float z_child = z;

        /* can we actually STORE this secondary? */
        int can_store = (next_free_per_batch != NULL) &&
                        (next_free_per_batch[l] < n_slots);

        if (can_store) {
          /* grab real slot */
          int child_i = next_free_per_batch[l];
          next_free_per_batch[l] += 1;

          particle_pusher(
            l, child_i,
            &x_child, &y_child, &z_child,
            vx_sec,
            &vy_sec, &vz_sec,
            px_sec, py_sec, pz_sec,
            mass_e, charge_e, gamma_sec,
            c, dt, n_steps - j,
            B, z0, B_sigma,
            layers, widths, angles, heights,
            ls0, ls1, hs0, hs1,
            ws0, ws1, as0, as1,
            n_layers, n_straws,
            /* real buffers */ response, trajectories,
            edep, r_mm, t0, hit_pos,
            rs0, rs1, rs2, rs3,
            trs0, trs1, trs2, trs3,
            step_offset + j,
            /* real mask */ mask, mask_s0, mask_s1,
            child_state, depth + 1, max_depth,
            p_spawn, E_sec_MeV,
      p_pair,
            next_free_per_batch,
      n_slots,
      sparse_buf
          );
        } else {
          /* array is full → simulate but DO NOT store anything */
          npy_float dummy_vy = vy_sec;
          npy_float dummy_vz = vz_sec;
          particle_pusher(
            l, i,               /* slot doesn't matter, we're not writing */
            &x_child, &y_child, &z_child,
            vx_sec,
            &dummy_vy, &dummy_vz,
            px_sec, py_sec, pz_sec,
            mass_e, charge_e, gamma_sec,
            c, dt, n_steps - j,
            B, z0, B_sigma,
            layers, widths, angles, heights,
            ls0, ls1, hs0, hs1,
            ws0, ws1, as0, as1,
            n_layers, n_straws,
            /* discard mode: NULL outputs */ NULL, NULL,
            NULL, NULL, NULL, NULL,
            0,0,0,0,
            0,0,0,0,
            step_offset + j,
            /* no mask */ NULL, 0, 0,
            child_state, depth + 1, max_depth,
            p_spawn, E_sec_MeV,
      0.0f,  /* p_pair */
            /* allocator unused */ NULL,
      0,
      NULL
          );
        }

        /* decorrelate parent RNG a bit */
  (void)xorshift32_next(rng_state);
  (void)xorshift32_next(rng_state);
}

/* -------------------------------------------------------------------------- */
/* particle pusher with recursive secondary spawn + discard when full         */
/* -------------------------------------------------------------------------- */
static void particle_pusher(
  /* identification */
  int l, int i,
  /* position/velocity pointers for in-place update */
  npy_float *x_ptr, npy_float *y_ptr, npy_float *z_ptr,
  const npy_float vx,
  npy_float *vy_ptr, npy_float *vz_ptr,
  /* primary momentum and mass/charge info */
  const npy_float px, const npy_float py, const npy_float pz,
  const npy_float mass, const npy_float charge, const npy_float gamma,
  /* integration params */
  const npy_float c, const npy_float dt, const int n_steps,
  /* B field params */
  const npy_float B, const npy_float z0, const npy_float B_sigma,
  /* geometry arrays + strides */
  const npy_float *layers, const npy_float *widths,
  const npy_float *angles, const npy_float *heights,
  const npy_intp ls0, const npy_intp ls1,
  const npy_intp hs0, const npy_intp hs1,
  const npy_intp ws0, const npy_intp ws1,
  const npy_intp as0, const npy_intp as1,
  /* detector sizes */
  const int n_layers, const int n_straws,
  /* outputs (may be NULL for discard mode) */
  npy_float *response, npy_float *trajectories,
  npy_float *edep, npy_float *r_mm, npy_float *t0, npy_float *hit_pos,
  /* response strides */
  const npy_intp rs0, const npy_intp rs1, const npy_intp rs2, const npy_intp rs3,
  /* trajectory strides */
  const npy_intp trs0, const npy_intp trs1, const npy_intp trs2, const npy_intp trs3,
  /* time offset (for secondaries) */
  int step_offset,
  /* mask info (may be NULL) */
  npy_float *mask, const npy_intp mask_s0, const npy_intp mask_s1,
  /* RNG + recursion */
  uint32_t rng_state, int depth, int max_depth,
  float p_spawn, float E_sec_MeV,
  float p_pair,  /* probability of e+e- pair production */
  /* slot allocator (may be NULL) */
  int *next_free_per_batch,
  int n_slots,
  /* sparse buffer (may be NULL, if provided, used instead of dense arrays) */
  sparse_hit_buffer_t *sparse_buf
) {
  npy_float x  = *x_ptr;
  npy_float y  = *y_ptr;
  npy_float z  = *z_ptr;
  npy_float vy = *vy_ptr;
  npy_float vz = *vz_ptr;

  /* secondaries get marked */
  if (mask && depth > 0) {
    mask[l * mask_s0 + i * mask_s1] = 1.0f;
  }

  uint32_t state = rng_state;

  for (int j = 0; j < n_steps; ++j) {
    /* magnetic-field kick */
    const npy_float Bx = B * expf(-square((z - z0) / B_sigma));
    const npy_float tx = c * Bx;
    const npy_float t_norm_sqr = tx * tx;

    const npy_float vy_m = vy + vz * tx;
    const npy_float vz_m = vz - vy * tx;
    const npy_float sx   = 2.0f * tx / (1.0f + t_norm_sqr);

    vy = vy_m + vz_m * sx;
    vz = vz_m - vy_m * sx;

    /* advance position (cm) */
    const npy_float dx = dt * vx * 29.9792f;
    const npy_float dy = dt * vy * 29.9792f;
    const npy_float dz = dt * vz * 29.9792f;

    const npy_float x_ = x + dx;
    const npy_float y_ = y + dy;
    const npy_float z_ = z + dz;

    /* --- layer / straw intersection --- */
    for (int k = 0; k < n_layers; ++k) {
      const npy_float layer  = layers[l * ls0 + k * ls1];
      const npy_float height = heights[l * hs0 + k * hs1];
      const npy_float width  = widths[l * ws0 + k * ws1];
      const npy_float r      = height / n_straws;
      const npy_float left   = layer - r;
      const npy_float right  = layer + r;

      if ((z < left && z_ < left) || (z > right && z_ > right)) {
        continue;
      }

      const npy_float angle = angles[l * as0 + k * as1];
      const npy_float nx = cosf(angle);
      const npy_float ny = sinf(angle);

      /* rotate to local frame */
      const npy_float ry = -ny * x + nx * y;
      const npy_float rx =  nx * x + ny * y;

      /* detector face as skewed quad */
      const npy_float skew = height * tanf(angle);
      npy_float corners[4][2] = {
        {-width - skew, -height},
        {-width + skew,  height},
        { width + skew,  height},
        { width - skew, -height}
      };

      if (!point_in_parallelogram(rx, ry, corners)) {
        continue;
      }

      const npy_int straw_i = (npy_int) floorf(0.5f * (ry + height) / r);
      const npy_float straw_y = (2 * straw_i + 1) * r - height;

      const npy_float sqr_distance_to_wire =
        square(z - layer) + square(ry - straw_y);

      if (sqr_distance_to_wire > r * r) {
        continue;
      }

      if (straw_i >= 0 && straw_i < n_straws) {
        /* calculate common values */
        npy_float r_mm_val = fabsf(ry - straw_y) * 10.0f;
        npy_float t0_val = (j + step_offset) * dt;
        npy_float edep_val = 0.0f;
        int is_first_hit = 0;

          /* Bethe–Bloch-ish dE/dx */
        if (edep || sparse_buf) {
            const npy_float K     = 0.307075f;   // MeV*cm^2/g
            const npy_float Z     = 18.0f;       // Argon
            const npy_float A     = 39.948f;     // Argon
            const npy_float I_exc = 188e-6f;     // MeV
            const npy_float rho   = 1.66e-3f;    // g/cm^3
            const npy_float me    = 0.511f;      // MeV/c^2

            npy_float p = sqrtf(px * px + py * py + pz * pz);
            npy_float beta = p / sqrtf(p * p + mass * mass);
            npy_float gamma_bethe = sqrtf(1.0f + (p * p) / (mass * mass));

            npy_float Tmax =
              (2.0f * me * beta * beta * gamma_bethe * gamma_bethe) /
              (1.0f + 2.0f * gamma_bethe * me / mass + (me / mass) * (me / mass));

            npy_float arg =
              (2.0f * me * beta * beta * gamma_bethe * gamma_bethe * Tmax) / (I_exc * I_exc);
            if (arg <= 0.0f) arg = 1e-10f;

            npy_float log_term = logf(arg);
            npy_float dEdx = K * (charge * charge) * Z / A / (beta * beta) *
              (0.5f * log_term - beta * beta) * rho;

            npy_float v = sqrtf(vx * vx + vy * vy + vz * vz);
            npy_float path_cm = v * dt * 29.9792f;

          edep_val = dEdx * path_cm;
        }

        if (sparse_buf) {
          /* sparse mode: check if hit already exists before adding */
          int hit_exists = 0;
          for (size_t idx = 0; idx < sparse_buf->count; ++idx) {
            if (sparse_buf->hits[idx].event == l &&
                sparse_buf->hits[idx].particle == i &&
                sparse_buf->hits[idx].layer == k &&
                sparse_buf->hits[idx].straw == straw_i) {
              hit_exists = 1;
              break;
            }
          }
          is_first_hit = !hit_exists;
          /* add to sparse buffer */
          sparse_buffer_add(sparse_buf, l, i, k, straw_i, dt, edep_val, r_mm_val, t0_val, x, y, z);
        } else if (response) {
          /* dense mode: use original logic */
          const npy_intp idx =
            l * rs0 + i * rs1 + k * rs2 + (npy_intp)straw_i * rs3;

          /* increment response by dt */
          response[idx] += dt;

          /* check if this is the first hit to this straw (for secondary spawning) */
          is_first_hit = (response[idx] == dt);

          /* record first hit position */
          if (hit_pos && is_first_hit) {
            hit_pos[idx * 3 + 0] = x;
            hit_pos[idx * 3 + 1] = y;
            hit_pos[idx * 3 + 2] = z;
          }

          /* record first hit time */
          if (t0 && is_first_hit) {
            t0[idx] = t0_val;
          }

          /* drift distance */
          if (r_mm) {
            r_mm[idx] = r_mm_val;
          }

          /* energy deposition */
          if (edep) {
            edep[idx] += edep_val;
          }
        }

        /* --- spawn secondary when particle hits straw tube --- */
        if (is_first_hit && depth < max_depth) {
          float r = rand01(&state);
          /* First check for pair production */
          if (p_pair > 0.0f && r < p_pair) {
            spawn_pair(
              l, i, x, y, z,
              px, py, pz, mass, charge, gamma,
              c, dt, n_steps, j,
              B, z0, B_sigma,
              layers, widths, angles, heights,
              ls0, ls1, hs0, hs1,
              ws0, ws1, as0, as1,
              n_layers, n_straws,
              response, trajectories,
              edep, r_mm, t0, hit_pos,
              rs0, rs1, rs2, rs3,
              trs0, trs1, trs2, trs3,
              step_offset,
              mask, mask_s0, mask_s1,
              &state, depth, max_depth,
              p_spawn, E_sec_MeV,
              p_pair,
              next_free_per_batch, n_slots,
              sparse_buf
            );
          }
          /* Otherwise check for single secondary */
          else if (p_spawn > 0.0f && r < (p_pair + p_spawn)) {
            spawn_secondary(
              l, i, x, y, z,
              px, py, pz, mass, charge, gamma,
              c, dt, n_steps, j,
              B, z0, B_sigma,
              layers, widths, angles, heights,
              ls0, ls1, hs0, hs1,
              ws0, ws1, as0, as1,
              n_layers, n_straws,
              response, trajectories,
              edep, r_mm, t0, hit_pos,
              rs0, rs1, rs2, rs3,
              trs0, trs1, trs2, trs3,
              step_offset,
              mask, mask_s0, mask_s1,
              &state, depth, max_depth,
              p_spawn, E_sec_MeV,
              p_pair,
              next_free_per_batch, n_slots,
              sparse_buf
            );
          }
        }
      }
    }

    /* write trajectory sample */
    x = x_;
    y = y_;
    z = z_;
    if (trajectories) {
      int tj = j + step_offset;
      trajectories[l * trs0 + i * trs1 + tj * trs2]               = x;
      trajectories[l * trs0 + i * trs1 + tj * trs2 + trs3]        = y;
      trajectories[l * trs0 + i * trs1 + tj * trs2 + 2 * trs3]    = z;
    }
  }

  *x_ptr  = x;
  *y_ptr  = y;
  *z_ptr  = z;
  *vy_ptr = vy;
  *vz_ptr = vz;
}

/* -------------------------------------------------------------------------- */
/* Python wrapper: solve                                                      */
/* -------------------------------------------------------------------------- */
static PyObject *solve(PyObject *self, PyObject *args) {
  /* python-level args */
  PyObject *py_dt = NULL;
  PyObject *py_B = NULL, *py_L = NULL;
  PyObject *py_initial_positions = NULL, *py_initial_momenta = NULL;
  PyObject *py_masses = NULL, *py_charges = NULL;
  PyObject *py_layers = NULL, *py_width = NULL, *py_heights = NULL, *py_angles = NULL;
  PyObject *py_z0 = NULL, *py_B_sigma = NULL;
  PyObject *py_steps = NULL;
  PyObject *py_trajectories = NULL, *py_response = NULL;
  PyObject *py_edep = NULL, *py_r_mm = NULL, *py_t0 = NULL, *py_hit_pos = NULL;
  PyObject *py_mask = NULL;

  if (!PyArg_UnpackTuple(
        args, "straw_solve", 21, 21,
        &py_initial_positions, &py_initial_momenta,
        &py_masses, &py_charges,
        &py_B, &py_L,
        &py_z0, &py_B_sigma,
        &py_steps, &py_dt,
        &py_layers, &py_width, &py_heights, &py_angles,
        &py_trajectories, &py_response, &py_edep, &py_r_mm, &py_t0, &py_hit_pos,
        &py_mask)) {
    return NULL;
  }

  /* basic checks */
  if (!PyLong_Check(py_steps)) {
    PyErr_SetString(PyExc_TypeError, "steps must be an int");
    return NULL;
  }
  const long n_steps = PyLong_AsLong(py_steps);

  if (!PyFloat_Check(py_dt)) {
    PyErr_SetString(PyExc_TypeError, "dt must be a float");
    return NULL;
  }
  const npy_float dt = (npy_float)PyFloat_AsDouble(py_dt);

  if (!PyArray_Check(py_response)) {
    PyErr_SetString(PyExc_TypeError, "response must be a numpy float32 array");
    return NULL;
  }
  const PyArrayObject *response_array = (PyArrayObject *)py_response;
  if (!(PyArray_TYPE(response_array) == NPY_FLOAT32 &&
        PyArray_NDIM(response_array) == 4)) {
    PyErr_SetString(PyExc_TypeError,
                    "response must be shape (batch, n_particles, n_layers, n_straws) float32");
    return NULL;
  }

  const npy_intp n_batch     = PyArray_DIM(response_array, 0);
  const npy_intp n_particles = PyArray_DIM(response_array, 1);
  const npy_intp n_layers    = PyArray_DIM(response_array, 2);
  const npy_intp n_straws    = PyArray_DIM(response_array, 3);

  /* optional trajectories */
  PyArrayObject *trajectories_array = NULL;
  if (!Py_IsNone(py_trajectories)) {
    if (!PyArray_Check(py_trajectories)) {
      PyErr_SetString(PyExc_TypeError, "trajectories must be a numpy array or None");
      return NULL;
    }
    trajectories_array = (PyArrayObject *)py_trajectories;
    if (!(PyArray_TYPE(trajectories_array) == NPY_FLOAT32 &&
          PyArray_NDIM(trajectories_array) == 4 &&
          PyArray_DIM(trajectories_array, 0) == n_batch &&
          PyArray_DIM(trajectories_array, 1) == n_particles &&
          PyArray_DIM(trajectories_array, 2) == n_steps &&
          PyArray_DIM(trajectories_array, 3) == SPACE_DIM)) {
      PyErr_SetString(PyExc_TypeError,
        "trajectories must be shape (batch, n_particles, n_steps, 3) float32");
      return NULL;
    }
  }

  /* input vectors/scalars */
  const PyArrayObject *initial_positions_array =
    check_vector_array(py_initial_positions, n_batch, n_particles);
  if (!initial_positions_array) {
    PyErr_SetString(PyExc_TypeError,
      "initial_positions must be shape (batch, n_particles, 3) float32");
    return NULL;
  }

  const PyArrayObject *initial_momenta_array =
    check_vector_array(py_initial_momenta, n_batch, n_particles);
  if (!initial_momenta_array) {
    PyErr_SetString(PyExc_TypeError,
      "initial_momenta must be shape (batch, n_particles, 3) float32");
    return NULL;
  }

  const PyArrayObject *masses_array =
    check_scalar_array(py_masses, n_batch, n_particles);
  if (!masses_array) {
    PyErr_SetString(PyExc_TypeError,
      "masses must be shape (batch, n_particles) float32");
    return NULL;
  }

  const PyArrayObject *charges_array =
    check_scalar_array(py_charges, n_batch, n_particles);
  if (!charges_array) {
    PyErr_SetString(PyExc_TypeError,
      "charges must be shape (batch, n_particles) float32");
    return NULL;
  }

  const PyArrayObject *layers_array =
    check_scalar_array(py_layers, n_batch, n_layers);
  if (!layers_array) {
    PyErr_SetString(PyExc_TypeError,
      "layers must be shape (batch, n_layers) float32");
    return NULL;
  }

  const PyArrayObject *width_array =
    check_scalar_array(py_width, n_batch, n_layers);
  if (!width_array) {
    PyErr_SetString(PyExc_TypeError,
      "widths must be shape (batch, n_layers) float32");
    return NULL;
  }

  const PyArrayObject *heights_array =
    check_scalar_array(py_heights, n_batch, n_layers);
  if (!heights_array) {
    PyErr_SetString(PyExc_TypeError,
      "heights must be shape (batch, n_layers) float32");
    return NULL;
  }

  const PyArrayObject *angles_array =
    check_scalar_array(py_angles, n_batch, n_layers);
  if (!angles_array) {
    PyErr_SetString(PyExc_TypeError,
      "angles must be shape (batch, n_layers) float32");
    return NULL;
  }

  const PyArrayObject *B_array  = check_design_array(py_B,  n_batch);
  const PyArrayObject *L_array  = check_design_array(py_L,  n_batch);
  const PyArrayObject *z0_array = check_design_array(py_z0, n_batch);
  const PyArrayObject *B_sigma_array = check_design_array(py_B_sigma, n_batch);
  if (!B_array || !L_array || !z0_array || !B_sigma_array) {
    PyErr_SetString(PyExc_TypeError,
      "B, L, z0, B_sigma must be shape (batch,) float32");
    return NULL;
  }

  /* optional outputs */
  npy_float *edep    = NULL;
  npy_float *r_mm    = NULL;
  npy_float *t0      = NULL;
  npy_float *hit_pos = NULL;

  if (py_edep && py_edep != Py_None) {
    edep = (npy_float *)PyArray_DATA((PyArrayObject *)py_edep);
  }
  if (py_r_mm && py_r_mm != Py_None) {
    r_mm = (npy_float *)PyArray_DATA((PyArrayObject *)py_r_mm);
  }
  if (py_t0 && py_t0 != Py_None) {
    t0 = (npy_float *)PyArray_DATA((PyArrayObject *)py_t0);
  }
  if (py_hit_pos && py_hit_pos != Py_None) {
    hit_pos = (npy_float *)PyArray_DATA((PyArrayObject *)py_hit_pos);
  }

  /* mask (new) */
  npy_float *mask = NULL;
  npy_intp  mask_s0 = 0, mask_s1 = 0;
  if (py_mask && py_mask != Py_None) {
    PyArrayObject *mask_array = (PyArrayObject *)py_mask;
    if (!(PyArray_TYPE(mask_array) == NPY_FLOAT32 &&
          PyArray_NDIM(mask_array) == 2 &&
          PyArray_DIM(mask_array, 0) == n_batch &&
          PyArray_DIM(mask_array, 1) == n_particles)) {
      PyErr_SetString(PyExc_TypeError,
        "mask must be shape (batch, n_particles) float32");
      return NULL;
    }
    mask = (npy_float *)PyArray_DATA(mask_array);
    mask_s0 = PyArray_STRIDE(mask_array, 0) / sizeof(npy_float);
    mask_s1 = PyArray_STRIDE(mask_array, 1) / sizeof(npy_float);
  }

  /* data pointers */
  const npy_float *initial_positions = (const npy_float *)PyArray_DATA(initial_positions_array);
  const npy_float *initial_momenta   = (const npy_float *)PyArray_DATA(initial_momenta_array);
  const npy_float *charges           = (const npy_float *)PyArray_DATA(charges_array);
  const npy_float *masses            = (const npy_float *)PyArray_DATA(masses_array);

  const npy_float *Bs       = (const npy_float *)PyArray_DATA(B_array);
  const npy_float *Ls       = (const npy_float *)PyArray_DATA(L_array);
  const npy_float *z0s      = (const npy_float *)PyArray_DATA(z0_array);
  const npy_float *B_sigmas = (const npy_float *)PyArray_DATA(B_sigma_array);

  const npy_float *layers = (const npy_float *)PyArray_DATA(layers_array);
  const npy_float *widths = (const npy_float *)PyArray_DATA(width_array);
  const npy_float *angles = (const npy_float *)PyArray_DATA(angles_array);
  const npy_float *heights= (const npy_float *)PyArray_DATA(heights_array);

  npy_float *response = (npy_float *)PyArray_DATA(response_array);
  npy_float *trajectories = trajectories_array
    ? (npy_float *)PyArray_DATA(trajectories_array)
    : NULL;

  /* strides */
  const npy_intp Bs0  = PyArray_STRIDE(B_array, 0)  / sizeof(npy_float);
  const npy_intp Ls0  = PyArray_STRIDE(L_array, 0)  / sizeof(npy_float);
  const npy_intp ips0 = PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  const npy_intp ips1 = PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);
  const npy_intp ips2 = PyArray_STRIDE(initial_positions_array, 2) / sizeof(npy_float);
  const npy_intp ivs0 = PyArray_STRIDE(initial_momenta_array, 0)   / sizeof(npy_float);
  const npy_intp ivs1 = PyArray_STRIDE(initial_momenta_array, 1)   / sizeof(npy_float);
  const npy_intp ivs2 = PyArray_STRIDE(initial_momenta_array, 2)   / sizeof(npy_float);
  const npy_intp chs0 = PyArray_STRIDE(charges_array, 0)           / sizeof(npy_float);
  const npy_intp chs1 = PyArray_STRIDE(charges_array, 1)           / sizeof(npy_float);
  const npy_intp ms0  = PyArray_STRIDE(masses_array, 0)            / sizeof(npy_float);
  const npy_intp ms1  = PyArray_STRIDE(masses_array, 1)            / sizeof(npy_float);
  const npy_intp ls0  = PyArray_STRIDE(layers_array, 0)            / sizeof(npy_float);
  const npy_intp ls1  = PyArray_STRIDE(layers_array, 1)            / sizeof(npy_float);
  const npy_intp hs0  = PyArray_STRIDE(heights_array, 0)           / sizeof(npy_float);
  const npy_intp hs1  = PyArray_STRIDE(heights_array, 1)           / sizeof(npy_float);
  const npy_intp ws0  = PyArray_STRIDE(width_array, 0)             / sizeof(npy_float);
  const npy_intp ws1  = PyArray_STRIDE(width_array, 1)             / sizeof(npy_float);
  const npy_intp as0  = PyArray_STRIDE(angles_array, 0)            / sizeof(npy_float);
  const npy_intp as1  = PyArray_STRIDE(angles_array, 1)            / sizeof(npy_float);
  const npy_intp rs0  = PyArray_STRIDE(response_array, 0)          / sizeof(npy_float);
  const npy_intp rs1  = PyArray_STRIDE(response_array, 1)          / sizeof(npy_float);
  const npy_intp rs2  = PyArray_STRIDE(response_array, 2)          / sizeof(npy_float);
  const npy_intp rs3  = PyArray_STRIDE(response_array, 3)          / sizeof(npy_float);
  const npy_intp trs0 = trajectories_array ? PyArray_STRIDE(trajectories_array, 0) / sizeof(npy_float) : 0;
  const npy_intp trs1 = trajectories_array ? PyArray_STRIDE(trajectories_array, 1) / sizeof(npy_float) : 0;
  const npy_intp trs2 = trajectories_array ? PyArray_STRIDE(trajectories_array, 2) / sizeof(npy_float) : 0;
  const npy_intp trs3 = trajectories_array ? PyArray_STRIDE(trajectories_array, 3) / sizeof(npy_float) : 0;

  /* ---------------------------------------------------------------------- */
  /* main loop                                                              */
  /* ---------------------------------------------------------------------- */

  /* per-batch "next free slot" */
  int *next_free = (int *)malloc(sizeof(int) * (size_t)n_batch);
  if (!next_free) {
    PyErr_SetString(PyExc_MemoryError, "failed to allocate next_free array");
    return NULL;
  }

  /* detect first free slot per batch: consider slots with ~zero momentum free */
  for (int l = 0; l < n_batch; ++l) {
    int first_free = (int)n_particles;  /* fallback: no free slot */
    for (int i = 0; i < (int)n_particles; ++i) {
      npy_float px = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];
      if (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW) {
        first_free = i;
        break;
      }
    }
    next_free[l] = first_free;
  }

  Py_BEGIN_ALLOW_THREADS

  for (int l = 0; l < n_batch; ++l) {
    const npy_float B = Bs[l * Bs0];
    const npy_float L = Ls[l * Ls0];
    (void)L;

    const npy_float z0_val      = z0s[l];
    const npy_float B_sigma_val = B_sigmas[l];

    for (int i = 0; i < n_particles; ++i) {
      /* initial state for this particle */
      npy_float x = initial_positions[l * ips0 + i * ips1];
      npy_float y = initial_positions[l * ips0 + i * ips1 + ips2];
      npy_float z = initial_positions[l * ips0 + i * ips1 + 2 * ips2];

      npy_float px = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];

      const npy_float charge = charges[l * chs0 + i * chs1];
      const npy_float mass   = masses[l * ms0  + i * ms1];

      /* if this is an empty slot, skip */
      if (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW) {
        continue;
      }

      npy_float p2 = px * px + py * py + pz * pz;
      const npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));

      npy_float vx = px / (gamma * mass);
      npy_float vy = py / (gamma * mass);
      npy_float vz = pz / (gamma * mass);

      const npy_float mass_MeV_kg = 1.78266192e-30f;
      const npy_float charge_e_C  = 1.602176634e-19f;
      const npy_float c =
        0.5f * (dt / 1e9f) * (charge * charge_e_C) / (mass * mass_MeV_kg) / gamma;

      /* deterministic per-particle RNG seed */
      uint32_t seed = RNG_SEED_BASE;
      seed ^= (uint32_t)l * 0x9e3779b1u;
      seed ^= (uint32_t)i * 0x85ebca6bu;
      seed ^= (uint32_t)((uintptr_t)&seed >> 3);

      particle_pusher(
        l, i,
        &x, &y, &z,
        vx,
        &vy, &vz,
        px, py, pz,
        mass, charge, gamma,
        c, dt, (int)n_steps,
        B, z0_val, B_sigma_val,
        layers, widths, angles, heights,
        ls0, ls1, hs0, hs1,
        ws0, ws1, as0, as1,
        (int)n_layers, (int)n_straws,
        response, trajectories,
        edep, r_mm, t0, hit_pos,
        rs0, rs1, rs2, rs3,
        trs0, trs1, trs2, trs3,
        /* step_offset */ 0,
        mask, mask_s0, mask_s1,
        seed,
        0, SEC_MAX_DEPTH,
        SEC_SPAWN_PROB, SEC_E_MEV,
        0.0f,  /* p_pair - can be made configurable later */
        /* slot allocator */
        next_free,
        (int)n_particles,
        /* sparse buffer: NULL for dense mode */
        NULL
      );
    }
  }

  Py_END_ALLOW_THREADS

  free(next_free);

  return PyLong_FromLong(0);
}

/* -------------------------------------------------------------------------- */
/* Python wrapper: solve_sparse - returns sparse hits as tuple of arrays     */
/* -------------------------------------------------------------------------- */
static PyObject *solve_sparse(PyObject *self, PyObject *args) {
  PyObject *py_dt = NULL;
  PyObject *py_B = NULL, *py_L = NULL;
  PyObject *py_initial_positions = NULL, *py_initial_momenta = NULL;
  PyObject *py_masses = NULL, *py_charges = NULL;
  PyObject *py_layers = NULL, *py_width = NULL, *py_heights = NULL, *py_angles = NULL;
  PyObject *py_z0 = NULL, *py_B_sigma = NULL;
  PyObject *py_steps = NULL;
  PyObject *py_trajectories = NULL;
  PyObject *py_mask = NULL;

  if (!PyArg_UnpackTuple(
        args, "straw_solve_sparse", 16, 16,
        &py_initial_positions, &py_initial_momenta,
        &py_masses, &py_charges,
        &py_B, &py_L,
        &py_z0, &py_B_sigma,
        &py_steps, &py_dt,
        &py_layers, &py_width, &py_heights, &py_angles,
        &py_trajectories, &py_mask)) {
    return NULL;
  }

  /* basic checks - similar to solve */
  if (!PyLong_Check(py_steps)) {
    PyErr_SetString(PyExc_TypeError, "steps must be an int");
    return NULL;
  }
  const long n_steps = PyLong_AsLong(py_steps);

  if (!PyFloat_Check(py_dt)) {
    PyErr_SetString(PyExc_TypeError, "dt must be a float");
    return NULL;
  }
  const npy_float dt = (npy_float)PyFloat_AsDouble(py_dt);

  /* get dimensions from initial_positions */
  const PyArrayObject *initial_positions_array =
    (PyArrayObject *)py_initial_positions;
  if (!PyArray_Check(initial_positions_array) ||
      PyArray_NDIM(initial_positions_array) != 3 ||
      PyArray_DIM(initial_positions_array, 2) != SPACE_DIM) {
    PyErr_SetString(PyExc_TypeError,
      "initial_positions must be shape (batch, n_particles, 3) float32");
    return NULL;
  }

  const npy_intp n_batch     = PyArray_DIM(initial_positions_array, 0);
  const npy_intp n_particles = PyArray_DIM(initial_positions_array, 1);

  /* get n_layers from layers array */
  const PyArrayObject *layers_array = (PyArrayObject *)py_layers;
  if (!PyArray_Check(layers_array) || PyArray_NDIM(layers_array) != 2) {
    PyErr_SetString(PyExc_TypeError, "layers must be 2D array");
    return NULL;
  }
  const npy_intp n_layers = PyArray_DIM(layers_array, 1);
  const npy_intp n_straws = 200; /* default, could be passed as param */

  /* validate other inputs */
  const PyArrayObject *initial_momenta_array =
    check_vector_array(py_initial_momenta, n_batch, n_particles);
  const PyArrayObject *masses_array =
    check_scalar_array(py_masses, n_batch, n_particles);
  const PyArrayObject *charges_array =
    check_scalar_array(py_charges, n_batch, n_particles);
  const PyArrayObject *width_array =
    check_scalar_array(py_width, n_batch, n_layers);
  const PyArrayObject *heights_array =
    check_scalar_array(py_heights, n_batch, n_layers);
  const PyArrayObject *angles_array =
    check_scalar_array(py_angles, n_batch, n_layers);
  const PyArrayObject *B_array  = check_design_array(py_B,  n_batch);
  const PyArrayObject *L_array  = check_design_array(py_L,  n_batch);
  const PyArrayObject *z0_array = check_design_array(py_z0, n_batch);
  const PyArrayObject *B_sigma_array = check_design_array(py_B_sigma, n_batch);

  if (!initial_momenta_array || !masses_array || !charges_array ||
      !width_array || !heights_array || !angles_array ||
      !B_array || !L_array || !z0_array || !B_sigma_array) {
    PyErr_SetString(PyExc_TypeError, "Invalid input array types/shapes");
    return NULL;
  }

  /* create sparse buffer: 100x the number of initial particles */
  size_t initial_capacity = (size_t)(n_particles * 100);
  sparse_hit_buffer_t *sparse_buf = sparse_buffer_create(initial_capacity);
  if (!sparse_buf) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate sparse buffer");
    return NULL;
  }

  /* get data pointers and strides (similar to solve) */
  const npy_float *initial_positions = (const npy_float *)PyArray_DATA(initial_positions_array);
  const npy_float *initial_momenta   = (const npy_float *)PyArray_DATA(initial_momenta_array);
  const npy_float *charges           = (const npy_float *)PyArray_DATA(charges_array);
  const npy_float *masses            = (const npy_float *)PyArray_DATA(masses_array);
  const npy_float *layers = (const npy_float *)PyArray_DATA(layers_array);
  const npy_float *widths = (const npy_float *)PyArray_DATA(width_array);
  const npy_float *angles = (const npy_float *)PyArray_DATA(angles_array);
  const npy_float *heights= (const npy_float *)PyArray_DATA(heights_array);
  const npy_float *Bs       = (const npy_float *)PyArray_DATA(B_array);
  const npy_float *Ls       = (const npy_float *)PyArray_DATA(L_array);
  const npy_float *z0s      = (const npy_float *)PyArray_DATA(z0_array);
  const npy_float *B_sigmas = (const npy_float *)PyArray_DATA(B_sigma_array);

  /* strides */
  const npy_intp ips0 = PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  const npy_intp ips1 = PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);
  const npy_intp ips2 = PyArray_STRIDE(initial_positions_array, 2) / sizeof(npy_float);
  const npy_intp ivs0 = PyArray_STRIDE(initial_momenta_array, 0)   / sizeof(npy_float);
  const npy_intp ivs1 = PyArray_STRIDE(initial_momenta_array, 1)   / sizeof(npy_float);
  const npy_intp ivs2 = PyArray_STRIDE(initial_momenta_array, 2)   / sizeof(npy_float);
  const npy_intp chs0 = PyArray_STRIDE(charges_array, 0)           / sizeof(npy_float);
  const npy_intp chs1 = PyArray_STRIDE(charges_array, 1)           / sizeof(npy_float);
  const npy_intp ms0  = PyArray_STRIDE(masses_array, 0)            / sizeof(npy_float);
  const npy_intp ms1  = PyArray_STRIDE(masses_array, 1)            / sizeof(npy_float);
  const npy_intp ls0  = PyArray_STRIDE(layers_array, 0)            / sizeof(npy_float);
  const npy_intp ls1  = PyArray_STRIDE(layers_array, 1)            / sizeof(npy_float);
  const npy_intp hs0  = PyArray_STRIDE(heights_array, 0)           / sizeof(npy_float);
  const npy_intp hs1  = PyArray_STRIDE(heights_array, 1)           / sizeof(npy_float);
  const npy_intp ws0  = PyArray_STRIDE(width_array, 0)             / sizeof(npy_float);
  const npy_intp ws1  = PyArray_STRIDE(width_array, 1)             / sizeof(npy_float);
  const npy_intp as0  = PyArray_STRIDE(angles_array, 0)            / sizeof(npy_float);
  const npy_intp as1  = PyArray_STRIDE(angles_array, 1)            / sizeof(npy_float);
  const npy_intp Bs0  = PyArray_STRIDE(B_array, 0)  / sizeof(npy_float);
  const npy_intp Ls0  = PyArray_STRIDE(L_array, 0)  / sizeof(npy_float);

  /* optional trajectories */
  PyArrayObject *trajectories_array = NULL;
  npy_float *trajectories = NULL;
  npy_intp trs0 = 0, trs1 = 0, trs2 = 0, trs3 = 0;
  if (!Py_IsNone(py_trajectories)) {
    if (!PyArray_Check(py_trajectories)) {
      PyErr_SetString(PyExc_TypeError, "trajectories must be a numpy array or None");
      sparse_buffer_free(sparse_buf);
      return NULL;
    }
    trajectories_array = (PyArrayObject *)py_trajectories;
    if (!(PyArray_TYPE(trajectories_array) == NPY_FLOAT32 &&
          PyArray_NDIM(trajectories_array) == 4 &&
          PyArray_DIM(trajectories_array, 0) == n_batch &&
          PyArray_DIM(trajectories_array, 1) == n_particles &&
          PyArray_DIM(trajectories_array, 2) == n_steps &&
          PyArray_DIM(trajectories_array, 3) == SPACE_DIM)) {
      PyErr_SetString(PyExc_TypeError,
        "trajectories must be shape (batch, n_particles, n_steps, 3) float32");
      sparse_buffer_free(sparse_buf);
      return NULL;
    }
    trajectories = (npy_float *)PyArray_DATA(trajectories_array);
    trs0 = PyArray_STRIDE(trajectories_array, 0) / sizeof(npy_float);
    trs1 = PyArray_STRIDE(trajectories_array, 1) / sizeof(npy_float);
    trs2 = PyArray_STRIDE(trajectories_array, 2) / sizeof(npy_float);
    trs3 = PyArray_STRIDE(trajectories_array, 3) / sizeof(npy_float);
  }

  /* mask */
  npy_float *mask = NULL;
  npy_intp  mask_s0 = 0, mask_s1 = 0;
  if (py_mask && py_mask != Py_None) {
    PyArrayObject *mask_array = (PyArrayObject *)py_mask;
    if (!(PyArray_TYPE(mask_array) == NPY_FLOAT32 &&
          PyArray_NDIM(mask_array) == 2 &&
          PyArray_DIM(mask_array, 0) == n_batch &&
          PyArray_DIM(mask_array, 1) == n_particles)) {
      PyErr_SetString(PyExc_TypeError,
        "mask must be shape (batch, n_particles) float32");
      sparse_buffer_free(sparse_buf);
      return NULL;
    }
    mask = (npy_float *)PyArray_DATA(mask_array);
    mask_s0 = PyArray_STRIDE(mask_array, 0) / sizeof(npy_float);
    mask_s1 = PyArray_STRIDE(mask_array, 1) / sizeof(npy_float);
  }

  /* per-batch "next free slot" */
  int *next_free = (int *)malloc(sizeof(int) * (size_t)n_batch);
  if (!next_free) {
    PyErr_SetString(PyExc_MemoryError, "failed to allocate next_free array");
    sparse_buffer_free(sparse_buf);
    return NULL;
  }

  /* detect first free slot per batch */
  for (int l = 0; l < n_batch; ++l) {
    int first_free = (int)n_particles;
    for (int i = 0; i < (int)n_particles; ++i) {
      npy_float px = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];
      if (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW) {
        first_free = i;
        break;
      }
    }
    next_free[l] = first_free;
  }

  Py_BEGIN_ALLOW_THREADS

  /* main loop - similar to solve but using sparse_buf */
  for (int l = 0; l < n_batch; ++l) {
    const npy_float B = Bs[l * Bs0];
    const npy_float L = Ls[l * Ls0];
    (void)L;

    const npy_float z0_val      = z0s[l];
    const npy_float B_sigma_val = B_sigmas[l];

    for (int i = 0; i < n_particles; ++i) {
      npy_float x = initial_positions[l * ips0 + i * ips1];
      npy_float y = initial_positions[l * ips0 + i * ips1 + ips2];
      npy_float z = initial_positions[l * ips0 + i * ips1 + 2 * ips2];

      npy_float px = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];

      const npy_float charge = charges[l * chs0 + i * chs1];
      const npy_float mass   = masses[l * ms0  + i * ms1];

      if (f32_abs(px) < SLOW && f32_abs(py) < SLOW && f32_abs(pz) < SLOW) {
        continue;
      }

      npy_float p2 = px * px + py * py + pz * pz;
      const npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));

      npy_float vx = px / (gamma * mass);
      npy_float vy = py / (gamma * mass);
      npy_float vz = pz / (gamma * mass);

      const npy_float mass_MeV_kg = 1.78266192e-30f;
      const npy_float charge_e_C  = 1.602176634e-19f;
      const npy_float c =
        0.5f * (dt / 1e9f) * (charge * charge_e_C) / (mass * mass_MeV_kg) / gamma;

      uint32_t seed = RNG_SEED_BASE;
      seed ^= (uint32_t)l * 0x9e3779b1u;
      seed ^= (uint32_t)i * 0x85ebca6bu;
      seed ^= (uint32_t)((uintptr_t)&seed >> 3);

      particle_pusher(
        l, i,
        &x, &y, &z,
        vx,
        &vy, &vz,
        px, py, pz,
        mass, charge, gamma,
        c, dt, (int)n_steps,
        B, z0_val, B_sigma_val,
        layers, widths, angles, heights,
        ls0, ls1, hs0, hs1,
        ws0, ws1, as0, as1,
        (int)n_layers, (int)n_straws,
        /* NULL for dense arrays - using sparse mode */ NULL, trajectories,
        NULL, NULL, NULL, NULL,
        0,0,0,0,
        trs0, trs1, trs2, trs3,
        0,
        mask, mask_s0, mask_s1,
        seed,
        0, SEC_MAX_DEPTH,
        SEC_SPAWN_PROB, SEC_E_MEV,
        0.0f,  /* p_pair - can be made configurable later */
        next_free,
        (int)n_particles,
        sparse_buf
      );
    }
  }

  Py_END_ALLOW_THREADS

  free(next_free);

  /* convert sparse hits to Python arrays */
  npy_intp n_hits = (npy_intp)sparse_buf->count;

  /* create separate arrays for each field */
  npy_intp dims[1] = {n_hits};

  PyArrayObject *events = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
  PyArrayObject *particles = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
  PyArrayObject *layers_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
  PyArrayObject *straws = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT32);
  PyArrayObject *values = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
  PyArrayObject *edep_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
  PyArrayObject *r_mm_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
  PyArrayObject *t0_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
  npy_intp hit_pos_dims[2] = {n_hits, 3};
  PyArrayObject *hit_pos_arr = (PyArrayObject *)PyArray_SimpleNew(2, hit_pos_dims, NPY_FLOAT32);

  if (!events || !particles || !layers_arr || !straws || !values ||
      !edep_arr || !r_mm_arr || !t0_arr || !hit_pos_arr) {
    Py_XDECREF(events); Py_XDECREF(particles); Py_XDECREF(layers_arr);
    Py_XDECREF(straws); Py_XDECREF(values); Py_XDECREF(edep_arr);
    Py_XDECREF(r_mm_arr); Py_XDECREF(t0_arr); Py_XDECREF(hit_pos_arr);
    PyErr_SetString(PyExc_MemoryError, "Failed to create result arrays");
    sparse_buffer_free(sparse_buf);
    return NULL;
  }

  /* copy data */
  int32_t *events_ptr = (int32_t *)PyArray_DATA(events);
  int32_t *particles_ptr = (int32_t *)PyArray_DATA(particles);
  int32_t *layers_ptr = (int32_t *)PyArray_DATA(layers_arr);
  int32_t *straws_ptr = (int32_t *)PyArray_DATA(straws);
  npy_float *values_ptr = (npy_float *)PyArray_DATA(values);
  npy_float *edep_ptr = (npy_float *)PyArray_DATA(edep_arr);
  npy_float *r_mm_ptr = (npy_float *)PyArray_DATA(r_mm_arr);
  npy_float *t0_ptr = (npy_float *)PyArray_DATA(t0_arr);
  npy_float *hit_pos_ptr = (npy_float *)PyArray_DATA(hit_pos_arr);

  for (size_t i = 0; i < sparse_buf->count; ++i) {
    const sparse_hit_t *hit = &sparse_buf->hits[i];
    events_ptr[i] = hit->event;
    particles_ptr[i] = hit->particle;
    layers_ptr[i] = hit->layer;
    straws_ptr[i] = hit->straw;
    values_ptr[i] = hit->value;
    edep_ptr[i] = hit->edep;
    r_mm_ptr[i] = hit->r_mm;
    t0_ptr[i] = hit->t0;
    hit_pos_ptr[i * 3 + 0] = hit->hit_pos[0];
    hit_pos_ptr[i * 3 + 1] = hit->hit_pos[1];
    hit_pos_ptr[i * 3 + 2] = hit->hit_pos[2];
  }

  sparse_buffer_free(sparse_buf);

  /* return as tuple of arrays */
  return Py_BuildValue("(OOOOOOOOO)", events, particles, layers_arr, straws,
                       values, edep_arr, r_mm_arr, t0_arr, hit_pos_arr);
}

/* -------------------------------------------------------------------------- */
/* module defs                                                                */
/* -------------------------------------------------------------------------- */
static PyMethodDef StrawDetectorMethods[] = {
  {"solve", solve, METH_VARARGS,
   "Solve trajectories and fill detector response, including secondaries."},
  {"solve_sparse", solve_sparse, METH_VARARGS,
   "Solve trajectories and return sparse hits as tuple of arrays: (events, particles, layers, straws, values, edep, r_mm, t0, hit_pos)."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef straw_detector_module = {
  PyModuleDef_HEAD_INIT,
  "straw_detector",
  NULL,
  -1,
  StrawDetectorMethods
};

PyMODINIT_FUNC
PyInit_straw_detector(void) {
  PyObject *m = PyModule_Create(&straw_detector_module);
  if (m == NULL) return NULL;

  import_array();

  return m;
}
