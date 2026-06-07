/*
 * straw_detector_cuda.cu
 *
 * CUDA port of straw_detector.c – same Python solve() interface.
 *
 * Thread mapping:
 *   grid  : (n_batch, ceil(n_particles / BLOCK_SIZE))
 *   block : (BLOCK_SIZE, 1)
 *   → each thread tracks one (event, primary_particle).
 *
 * Secondaries are handled iteratively via a per-thread task stack
 * (no GPU recursion).  Duplicate-hit detection is local to the thread
 * rather than scanning the shared sparse array.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

/* ── constants ───────────────────────────────────────────────── */
#define SPACE_DIM        3
#define SLOW             1.0e-6f
#define RNG_SEED_BASE    123456789u
#define MAX_LOCAL_HITS   128   /* max hit records per particle (per thread) */
#define MAX_TASK_STACK    32   /* max secondary-task stack depth             */
#define BLOCK_SIZE       128   /* threads per block                          */

#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif

/* ── CUDA error helper ───────────────────────────────────────── */
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
      PyErr_Format(PyExc_RuntimeError, "CUDA error at %s:%d – %s",         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));             \
      return NULL;                                                          \
    }                                                                       \
  } while (0)

#define CUDA_CHECK_GOTO(call, label)                                        \
  do {                                                                      \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
      PyErr_Format(PyExc_RuntimeError, "CUDA error at %s:%d – %s",         \
                   __FILE__, __LINE__, cudaGetErrorString(_e));             \
      goto label;                                                           \
    }                                                                       \
  } while (0)

/* ── device helpers ──────────────────────────────────────────── */

__device__ static inline uint32_t xorshift32(uint32_t *s) {
  uint32_t x = *s;
  x ^= x << 13; x ^= x >> 17; x ^= x << 5;
  *s = x ? x : 0xdeadbeefu;
  return *s;
}

__device__ static inline float rand01(uint32_t *s) {
  return (float)(xorshift32(s) >> 8) * (1.0f / 16777216.0f);
}

__device__ static inline int point_in_parallelogram(
    float x, float y, const float corners[4][2])
{
  int sign = 0;
  for (int i = 0; i < 4; ++i) {
    int j = (i + 1) & 3;
    float dx    = corners[j][0] - corners[i][0];
    float dy    = corners[j][1] - corners[i][1];
    float cross = dx * (y - corners[i][1]) - dy * (x - corners[i][0]);
    if (cross == 0.0f) continue;
    int s = (cross > 0.0f) ? 1 : -1;
    if (sign == 0) sign = s;
    else if (s != sign) return 0;
  }
  return 1;
}

/* ── particle task (for iterative secondary tracking) ────────── */

struct ParticleTask {
  float x0, y0, z0;
  float px0, py0, pz0;
  float mass, charge, t_initial;
  int   start_step, depth, particle_idx;
};

/* ── main tracking kernel ────────────────────────────────────── */

__global__ void track_kernel(
    /* dimensions */
    int n_particles, int n_steps, int n_layers, int n_straws,
    float dt, int max_particles, int max_depth,
    float p_spawn_single, float p_spawn_pair, float E_sec_MeV,
    /* inputs  [batch-major, C-contiguous] */
    const float * __restrict__ initial_positions, /* [B, P, 3] */
    const float * __restrict__ initial_momenta,   /* [B, P, 3] */
    const float * __restrict__ masses,            /* [B, P]    */
    const float * __restrict__ charges,           /* [B, P]    */
    const float * __restrict__ initial_times,     /* [B, P]    */
    const float * __restrict__ Bs,                /* [B]       */
    const float * __restrict__ z0s,               /* [B]       */
    const float * __restrict__ B_sigmas,          /* [B]       */
    const float * __restrict__ layers,            /* [B, L]    */
    const float * __restrict__ heights,           /* [B, L]    */
    const float * __restrict__ widths,            /* [B, L]    */
    const float * __restrict__ angles,            /* [B, L]    */
    /* sparse output */
    int   * __restrict__ sparse_events,
    int   * __restrict__ sparse_particles,
    int   * __restrict__ sparse_layers_arr,
    int   * __restrict__ sparse_straws_arr,
    float * __restrict__ sparse_values,
    float * __restrict__ sparse_r_mm,
    float * __restrict__ sparse_t0,
    float * __restrict__ sparse_hit_pos,
    int   * __restrict__ sparse_count,   /* global atomic counter */
    int    sparse_capacity,              /* max entries in sparse arrays */
    /* optional trajectory buffer */
    float * __restrict__ trajectories,   /* [B, P, T, 3] or NULL */
    /* per-event secondary index counter [B], initialised on host */
    int   * __restrict__ d_next_sec_idx)
{
  const int event_idx   = blockIdx.x;
  const int primary_idx = (int)blockIdx.y * blockDim.x + (int)threadIdx.x;
  if (primary_idx >= n_particles) return;

  /* ── initialise task stack with this primary ── */
  ParticleTask task_stack[MAX_TASK_STACK];
  int stack_top = 0;

  {
    float mass   = masses  [event_idx * n_particles + primary_idx];
    float charge = charges [event_idx * n_particles + primary_idx];
    if (mass <= SLOW) return;

    ParticleTask t;
    int base3 = (event_idx * n_particles + primary_idx) * 3;
    t.x0  = initial_positions[base3 + 0];
    t.y0  = initial_positions[base3 + 1];
    t.z0  = initial_positions[base3 + 2];
    t.px0 = initial_momenta  [base3 + 0];
    t.py0 = initial_momenta  [base3 + 1];
    t.pz0 = initial_momenta  [base3 + 2];
    t.mass         = mass;
    t.charge       = charge;
    t.t_initial    = initial_times[event_idx * n_particles + primary_idx];
    t.start_step   = 0;
    t.depth        = 0;
    t.particle_idx = primary_idx;
    task_stack[stack_top++] = t;
  }

  /* per-event constants */
  const float B        = Bs      [event_idx];
  const float z0_field = z0s     [event_idx];
  const float B_sigma  = B_sigmas[event_idx];

  /* ── local hit dedup (per thread; indexed by particle within this tree) ── */
  int local_hit_part  [MAX_LOCAL_HITS];
  int local_hit_layer [MAX_LOCAL_HITS];
  int local_hit_straw [MAX_LOCAL_HITS];
  int local_hit_count = 0;

  /* ── process task stack ── */
  while (stack_top > 0) {
    ParticleTask task = task_stack[--stack_top];

    const float mass   = task.mass;
    const float charge = task.charge;
    const int   pidx   = task.particle_idx;
    const int   depth  = task.depth;

    if (mass <= SLOW) continue;
    if (fabsf(task.px0) < SLOW && fabsf(task.py0) < SLOW && fabsf(task.pz0) < SLOW)
      continue;

    float x = task.x0, y = task.y0, z = task.z0;
    float px = task.px0, py = task.py0, pz = task.pz0;

    const float p2    = px*px + py*py + pz*pz;
    const float gamma = sqrtf(1.0f + p2 / (mass * mass));
    float vx = px / (gamma * mass);
    float vy = py / (gamma * mass);
    float vz = pz / (gamma * mass);

    const float mass_MeV_kg = 1.78266192e-30f;
    const float charge_e_C  = 1.602176634e-19f;
    const float c_boris = 0.5f * (dt / 1e9f) * (charge * charge_e_C) /
                          (mass * mass_MeV_kg) / gamma;

    uint32_t rng = RNG_SEED_BASE +
                   (uint32_t)(event_idx * 10000 + pidx + depth * 1000);

    /* ── step loop ── */
    for (int j = task.start_step; j < n_steps; ++j) {
      const float dz_norm  = (z - z0_field) / B_sigma;
      const float Bfield   = B * expf(-dz_norm * dz_norm);
      const float tx       = c_boris * Bfield;
      const float t2       = tx * tx;

      /* Boris rotation */
      const float vy_m = vy + vz * tx;
      const float vz_m = vz - vy * tx;
      const float sx   = 2.0f * tx / (1.0f + t2);
      vy = vy_m + vz_m * sx;
      vz = vz_m - vy_m * sx;

      const float x_ = x + dt * vx * 29.9792f;
      const float y_ = y + dt * vy * 29.9792f;
      const float z_ = z + dt * vz * 29.9792f;

      /* ── layer loop ── */
      for (int k = 0; k < n_layers; ++k) {
        const float layer  = layers [event_idx * n_layers + k];
        const float height = heights[event_idx * n_layers + k];
        const float width  = widths [event_idx * n_layers + k];
        const float r      = height / (float)n_straws;
        const float half_t = r;
        const float left   = layer - half_t;
        const float right  = layer + half_t;

        /* fast reject */
        if ((z < left && z_ < left) || (z > right && z_ > right)) continue;

        /* rotate to layer frame */
        const float angle = angles[event_idx * n_layers + k];
        const float ca = cosf(angle), sa = sinf(angle);
        const float rx  =  ca * x  + sa * y;
        const float ry  = -sa * x  + ca * y;

        float corners[4][2] = {
          {-width, -height}, {-width,  height},
          { width,  height}, { width, -height}
        };
        if (!point_in_parallelogram(rx, ry, corners)) continue;

        const int   straw_center = (int)floorf(0.5f * (ry + height) / r);
        const float rx_ =  ca * x_ + sa * y_;
        const float ry_ = -sa * x_ + ca * y_;
        const float drx = rx_ - rx;
        const float dry = ry_ - ry;
        const float dz_w = z_ - z;

        int   best_straw = -1;
        float best_dist2 = r * r + 1.0f;

        for (int off = -2; off <= 2; ++off) {
          const int si = straw_center + off;
          if (si < 0 || si >= n_straws) continue;

          const float straw_y = (2 * si + 1) * r - height;
          /* AC = wire_point - particle_start */
          const float acy = straw_y - ry;
          const float acz = layer   - z;
          /* cross product AB × (1,0,0) */
          const float cy  = dz_w;
          const float cz  = -dry;
          const float cn  = sqrtf(cy*cy + cz*cz);

          float dist2;
          if (cn < 1e-6f) {
            dist2 = acy*acy + acz*acz;
          } else {
            float dot  = acy * cy + acz * cz;
            float dist = fabsf(dot) / cn;
            dist2 = dist * dist;
          }

          if (dist2 < r*r && dist2 < best_dist2) {
            best_straw = si;
            best_dist2 = dist2;
          }
        }

        if (best_straw < 0) continue;

        /* duplicate check – local to this thread */
        int is_first = 1;
        for (int h = 0; h < local_hit_count; ++h) {
          if (local_hit_part [h] == pidx &&
              local_hit_layer[h] == k    &&
              local_hit_straw[h] == best_straw) {
            is_first = 0; break;
          }
        }
        if (!is_first) continue;

        if (local_hit_count < MAX_LOCAL_HITS) {
          local_hit_part [local_hit_count] = pidx;
          local_hit_layer[local_hit_count] = k;
          local_hit_straw[local_hit_count] = best_straw;
          ++local_hit_count;
        }

        /* claim a slot in the global sparse array */
        int si = atomicAdd(sparse_count, 1);
        if (si < sparse_capacity) {
          sparse_events    [si]     = event_idx;
          sparse_particles [si]     = pidx;
          sparse_layers_arr[si]     = k;
          sparse_straws_arr[si]     = best_straw;
          sparse_values    [si]     = dt;
          sparse_r_mm      [si]     = sqrtf(best_dist2) * 10.0f;
          sparse_t0        [si]     = task.t_initial + j * dt;
          sparse_hit_pos   [si*3+0] = x;
          sparse_hit_pos   [si*3+1] = y;
          sparse_hit_pos   [si*3+2] = z;
        }

        /* ── spawn secondaries ── */
        if (depth < max_depth &&
            (p_spawn_single > 0.0f || p_spawn_pair > 0.0f)) {
          float r_spawn = rand01(&rng);
          const float mass_e = 0.511f;

          if (r_spawn < p_spawn_single) {
            /* single electron */
            int e_idx = atomicAdd(&d_next_sec_idx[event_idx], 1);
            if (e_idx < max_particles && stack_top < MAX_TASK_STACK - 1) {
              float u = rand01(&rng), v = rand01(&rng);
              float cos_t = 2.0f * u - 1.0f;
              float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
              float phi   = 2.0f * M_PIf * v;
              float p_sec = sqrtf(E_sec_MeV*E_sec_MeV + 2.0f*E_sec_MeV*mass_e);
              ParticleTask sec;
              sec.x0 = x_; sec.y0 = y_; sec.z0 = z_;
              sec.px0 = p_sec * sin_t * cosf(phi);
              sec.py0 = p_sec * sin_t * sinf(phi);
              sec.pz0 = p_sec * cos_t;
              sec.mass = mass_e; sec.charge = -1.0f;
              sec.t_initial  = task.t_initial;
              sec.start_step = j + 1;
              sec.depth      = depth + 1;
              sec.particle_idx = e_idx;
              task_stack[stack_top++] = sec;
            }

          } else if (r_spawn < p_spawn_single + p_spawn_pair) {
            /* e+e- pair */
            int e_minus = atomicAdd(&d_next_sec_idx[event_idx], 1);
            int e_plus  = atomicAdd(&d_next_sec_idx[event_idx], 1);
            if (e_minus < max_particles && e_plus < max_particles &&
                stack_top < MAX_TASK_STACK - 2) {
              float u   = rand01(&rng), v = rand01(&rng);
              float cos_t = 2.0f * u - 1.0f;
              float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
              float phi   = 2.0f * M_PIf * v;
              float T_h   = E_sec_MeV * 0.5f;
              float p_sec = sqrtf(T_h*T_h + 2.0f*T_h*mass_e);
              float ppx = p_sec * sin_t * cosf(phi);
              float ppy = p_sec * sin_t * sinf(phi);
              float ppz = p_sec * cos_t;

              ParticleTask em;
              em.x0=x; em.y0=y; em.z0=z;
              em.px0=ppx; em.py0=ppy; em.pz0=ppz;
              em.mass=mass_e; em.charge=-1.0f;
              em.t_initial=task.t_initial;
              em.start_step=j; em.depth=depth+1; em.particle_idx=e_minus;

              ParticleTask ep;
              ep.x0=x; ep.y0=y; ep.z0=z;
              ep.px0=-ppx; ep.py0=-ppy; ep.pz0=-ppz;
              ep.mass=mass_e; ep.charge=+1.0f;
              ep.t_initial=task.t_initial;
              ep.start_step=j; ep.depth=depth+1; ep.particle_idx=e_plus;

              task_stack[stack_top++] = em;
              task_stack[stack_top++] = ep;
            }
          }
        } /* end secondary spawn */
      } /* end layer loop */

      x = x_; y = y_; z = z_;

      /* store trajectory */
      if (trajectories != NULL && pidx < n_particles) {
        int tbase = ((event_idx * n_particles + pidx) * n_steps + j) * 3;
        trajectories[tbase + 0] = x;
        trajectories[tbase + 1] = y;
        trajectories[tbase + 2] = z;
      }
    } /* end step loop */
  } /* end task stack */
}

/* ══════════════════════════════════════════════════════════════
 *  Python wrapper – same argument list as straw_detector.c
 * ══════════════════════════════════════════════════════════════ */

/* Convenience: get C-contiguous float32 array from a Python object. */
static PyArrayObject *get_contiguous_f32(PyObject *obj) {
  if (!PyArray_Check(obj)) return NULL;
  PyArrayObject *arr = (PyArrayObject *)obj;
  if (PyArray_TYPE(arr) != NPY_FLOAT32) return NULL;
  return (PyArrayObject *)PyArray_GETCONTIGUOUS(arr);
}

static PyObject *solve(PyObject *self, PyObject *args) {
  PyObject *py_initial_positions = NULL, *py_initial_momenta = NULL;
  PyObject *py_masses = NULL,  *py_charges = NULL, *py_initial_times = NULL;
  PyObject *py_B = NULL, *py_z0 = NULL, *py_B_sigma = NULL;
  PyObject *py_steps = NULL, *py_dt = NULL;
  PyObject *py_n_batch = NULL, *py_n_particles = NULL;
  PyObject *py_n_layers = NULL, *py_n_straws = NULL;
  PyObject *py_layers = NULL, *py_width = NULL;
  PyObject *py_heights = NULL, *py_angles = NULL;
  PyObject *py_trajectories = NULL;
  PyObject *py_sparse_events = NULL, *py_sparse_particles = NULL;
  PyObject *py_sparse_layers = NULL, *py_sparse_straws = NULL;
  PyObject *py_sparse_values = NULL, *py_sparse_r_mm = NULL;
  PyObject *py_sparse_t0 = NULL, *py_sparse_hit_pos = NULL;
  PyObject *py_sparse_count = NULL;
  PyObject *py_p_spawn_single = NULL, *py_p_spawn_pair = NULL;
  PyObject *py_E_sec = NULL, *py_max_particles = NULL;

  if (!PyArg_UnpackTuple(
          args, "straw_solve", 32, 32,
          &py_initial_positions, &py_initial_momenta,
          &py_masses, &py_charges, &py_initial_times,
          &py_B, &py_z0, &py_B_sigma,
          &py_steps, &py_dt,
          &py_n_batch, &py_n_particles,
          &py_n_layers, &py_n_straws,
          &py_layers, &py_width, &py_heights, &py_angles,
          &py_trajectories,
          &py_sparse_events, &py_sparse_particles,
          &py_sparse_layers, &py_sparse_straws,
          &py_sparse_values, &py_sparse_r_mm,
          &py_sparse_t0, &py_sparse_hit_pos, &py_sparse_count,
          &py_p_spawn_single, &py_p_spawn_pair,
          &py_E_sec, &py_max_particles))
    return NULL;

  /* scalar params */
  if (!PyLong_Check(py_steps) || !PyLong_Check(py_n_batch) ||
      !PyLong_Check(py_n_particles) || !PyLong_Check(py_n_layers) ||
      !PyLong_Check(py_n_straws) || !PyLong_Check(py_max_particles)) {
    PyErr_SetString(PyExc_TypeError, "steps/n_batch/n_particles/n_layers/n_straws/max_particles must be ints");
    return NULL;
  }
  if (!PyFloat_Check(py_dt) || !PyFloat_Check(py_p_spawn_single) ||
      !PyFloat_Check(py_p_spawn_pair) || !PyFloat_Check(py_E_sec)) {
    PyErr_SetString(PyExc_TypeError, "dt/p_spawn_single/p_spawn_pair/E_sec must be floats");
    return NULL;
  }

  const int n_steps       = (int)PyLong_AsLong(py_steps);
  const int n_batch       = (int)PyLong_AsLong(py_n_batch);
  const int n_particles   = (int)PyLong_AsLong(py_n_particles);
  const int n_layers      = (int)PyLong_AsLong(py_n_layers);
  const int n_straws      = (int)PyLong_AsLong(py_n_straws);
  const int max_particles = (int)PyLong_AsLong(py_max_particles);
  const float dt          = (float)PyFloat_AsDouble(py_dt);
  const float p_spawn_s   = (float)PyFloat_AsDouble(py_p_spawn_single);
  const float p_spawn_p   = (float)PyFloat_AsDouble(py_p_spawn_pair);
  const float E_sec_MeV   = (float)PyFloat_AsDouble(py_E_sec);
  const int   max_depth   = 2; /* matches original call-site */

  /* get contiguous float32 copies (DECREF after use) */
#define GET_ARR(name, obj)                                          \
  PyArrayObject *name##_arr = get_contiguous_f32(obj);             \
  if (!(name##_arr)) {                                              \
    PyErr_SetString(PyExc_TypeError,                                \
                    "Array " #name " must be float32");             \
    return NULL;                                                    \
  }

  GET_ARR(pos,      py_initial_positions)
  GET_ARR(mom,      py_initial_momenta)
  GET_ARR(mass,     py_masses)
  GET_ARR(charge,   py_charges)
  GET_ARR(itime,    py_initial_times)
  GET_ARR(B,        py_B)
  GET_ARR(z0,       py_z0)
  GET_ARR(Bsig,     py_B_sigma)
  GET_ARR(layers,   py_layers)
  GET_ARR(width,    py_width)
  GET_ARR(heights,  py_heights)
  GET_ARR(angles,   py_angles)
#undef GET_ARR

  /* sparse output arrays (int32 / float32 from Python) */
  PyArrayObject *sp_ev_arr   = (PyArrayObject *)py_sparse_events;
  PyArrayObject *sp_pa_arr   = (PyArrayObject *)py_sparse_particles;
  PyArrayObject *sp_la_arr   = (PyArrayObject *)py_sparse_layers;
  PyArrayObject *sp_st_arr   = (PyArrayObject *)py_sparse_straws;
  PyArrayObject *sp_va_arr   = (PyArrayObject *)py_sparse_values;
  PyArrayObject *sp_rm_arr   = (PyArrayObject *)py_sparse_r_mm;
  PyArrayObject *sp_t0_arr   = (PyArrayObject *)py_sparse_t0;
  PyArrayObject *sp_hp_arr   = (PyArrayObject *)py_sparse_hit_pos;
  PyArrayObject *sp_cnt_arr  = (PyArrayObject *)py_sparse_count;

  const int sparse_capacity = (int)PyArray_SIZE(sp_ev_arr);

  /* optional trajectories */
  PyArrayObject *traj_arr = NULL;
  if (!Py_IsNone(py_trajectories)) {
    if (!PyArray_Check(py_trajectories) ||
        PyArray_TYPE((PyArrayObject *)py_trajectories) != NPY_FLOAT32) {
      PyErr_SetString(PyExc_TypeError, "trajectories must be float32 or None");
      goto cleanup_arrs;
    }
    traj_arr = (PyArrayObject *)PyArray_GETCONTIGUOUS(
                   (PyArrayObject *)py_trajectories);
  }

  /* ── compute n_primaries per event (for d_next_sec_idx init) ── */
  {
    const float *h_masses = (const float *)PyArray_DATA(mass_arr);
    int *h_next_sec = (int *)malloc(n_batch * sizeof(int));
    if (!h_next_sec) { PyErr_NoMemory(); goto cleanup_arrs; }
    for (int l = 0; l < n_batch; ++l) {
      int np = 0;
      for (int i = 0; i < n_particles; ++i) {
        if (h_masses[l * n_particles + i] > SLOW) np = i + 1;
      }
      h_next_sec[l] = np;
    }

    /* ── device buffers ── */
    float *d_pos=NULL, *d_mom=NULL, *d_mass=NULL, *d_charge=NULL;
    float *d_itime=NULL, *d_Bs=NULL, *d_z0s=NULL, *d_Bsig=NULL;
    float *d_layers=NULL, *d_heights=NULL, *d_widths=NULL, *d_angles=NULL;
    int   *d_sp_ev=NULL, *d_sp_pa=NULL, *d_sp_la=NULL, *d_sp_st=NULL;
    float *d_sp_va=NULL, *d_sp_rm=NULL, *d_sp_t0=NULL, *d_sp_hp=NULL;
    int   *d_sp_cnt=NULL;
    float *d_traj=NULL;
    int   *d_next_sec=NULL;

    /* declare launch params here so goto can't jump over them */
    int grid_y = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(n_batch, grid_y);
    dim3 block(BLOCK_SIZE, 1);

    PyObject *ret = NULL;

#define ALLOC_COPY(dptr, harr, T)                                          \
  do {                                                                     \
    size_t _sz = (size_t)PyArray_NBYTES(harr);                            \
    CUDA_CHECK_GOTO(cudaMalloc((void**)&(dptr), _sz), cleanup_gpu);       \
    CUDA_CHECK_GOTO(cudaMemcpy((dptr), PyArray_DATA(harr),                \
                               _sz, cudaMemcpyHostToDevice), cleanup_gpu); \
  } while(0)

    ALLOC_COPY(d_pos,     pos_arr,    float);
    ALLOC_COPY(d_mom,     mom_arr,    float);
    ALLOC_COPY(d_mass,    mass_arr,   float);
    ALLOC_COPY(d_charge,  charge_arr, float);
    ALLOC_COPY(d_itime,   itime_arr,  float);
    ALLOC_COPY(d_Bs,      B_arr,      float);
    ALLOC_COPY(d_z0s,     z0_arr,     float);
    ALLOC_COPY(d_Bsig,    Bsig_arr,   float);
    ALLOC_COPY(d_layers,  layers_arr, float);
    ALLOC_COPY(d_heights, heights_arr,float);
    ALLOC_COPY(d_widths,  width_arr,  float);
    ALLOC_COPY(d_angles,  angles_arr, float);
#undef ALLOC_COPY

    /* sparse output (device) – write-only, no need to copy h→d */
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_ev,  sparse_capacity * sizeof(int)),   cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_pa,  sparse_capacity * sizeof(int)),   cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_la,  sparse_capacity * sizeof(int)),   cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_st,  sparse_capacity * sizeof(int)),   cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_va,  sparse_capacity * sizeof(float)), cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_rm,  sparse_capacity * sizeof(float)), cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_t0,  sparse_capacity * sizeof(float)), cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_hp,  sparse_capacity * 3 * sizeof(float)), cleanup_gpu);

    /* sparse counter: single int, init to 0 */
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_sp_cnt, sizeof(int)), cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMemset(d_sp_cnt, 0, sizeof(int)),       cleanup_gpu);

    /* per-event secondary counter */
    CUDA_CHECK_GOTO(cudaMalloc((void**)&d_next_sec, n_batch * sizeof(int)), cleanup_gpu);
    CUDA_CHECK_GOTO(cudaMemcpy(d_next_sec, h_next_sec, n_batch * sizeof(int),
                               cudaMemcpyHostToDevice), cleanup_gpu);

    /* trajectory buffer */
    if (traj_arr) {
      size_t tsz = (size_t)PyArray_NBYTES(traj_arr);
      CUDA_CHECK_GOTO(cudaMalloc((void**)&d_traj, tsz), cleanup_gpu);
      CUDA_CHECK_GOTO(cudaMemset(d_traj, 0, tsz),       cleanup_gpu);
    }

    /* ── launch ── */
    Py_BEGIN_ALLOW_THREADS

    track_kernel<<<grid, block>>>(
        n_particles, n_steps, n_layers, n_straws,
        dt, max_particles, max_depth,
        p_spawn_s, p_spawn_p, E_sec_MeV,
        d_pos, d_mom, d_mass, d_charge, d_itime,
        d_Bs, d_z0s, d_Bsig,
        d_layers, d_heights, d_widths, d_angles,
        d_sp_ev, d_sp_pa, d_sp_la, d_sp_st,
        d_sp_va, d_sp_rm, d_sp_t0, d_sp_hp,
        d_sp_cnt, sparse_capacity,
        d_traj, d_next_sec);

    cudaDeviceSynchronize();

    Py_END_ALLOW_THREADS

    if (cudaGetLastError() != cudaSuccess) {
      PyErr_Format(PyExc_RuntimeError, "CUDA kernel failed: %s",
                   cudaGetErrorString(cudaGetLastError()));
      goto cleanup_gpu;
    }

    /* ── copy sparse output d→h ── */
#define COPY_BACK(dptr, harr)                                              \
  CUDA_CHECK_GOTO(cudaMemcpy(PyArray_DATA(harr), (dptr),                  \
                             (size_t)PyArray_NBYTES(harr),                 \
                             cudaMemcpyDeviceToHost), cleanup_gpu)

    COPY_BACK(d_sp_ev,  sp_ev_arr);
    COPY_BACK(d_sp_pa,  sp_pa_arr);
    COPY_BACK(d_sp_la,  sp_la_arr);
    COPY_BACK(d_sp_st,  sp_st_arr);
    COPY_BACK(d_sp_va,  sp_va_arr);
    COPY_BACK(d_sp_rm,  sp_rm_arr);
    COPY_BACK(d_sp_t0,  sp_t0_arr);
    COPY_BACK(d_sp_hp,  sp_hp_arr);
    COPY_BACK(d_sp_cnt, sp_cnt_arr);
#undef COPY_BACK

    if (traj_arr)
      CUDA_CHECK_GOTO(cudaMemcpy(PyArray_DATA(traj_arr), d_traj,
                                 (size_t)PyArray_NBYTES(traj_arr),
                                 cudaMemcpyDeviceToHost), cleanup_gpu);

    ret = PyLong_FromLong(0);

  cleanup_gpu:
    cudaFree(d_pos);     cudaFree(d_mom);    cudaFree(d_mass);
    cudaFree(d_charge);  cudaFree(d_itime);  cudaFree(d_Bs);
    cudaFree(d_z0s);     cudaFree(d_Bsig);   cudaFree(d_layers);
    cudaFree(d_heights); cudaFree(d_widths); cudaFree(d_angles);
    cudaFree(d_sp_ev);   cudaFree(d_sp_pa);  cudaFree(d_sp_la);
    cudaFree(d_sp_st);   cudaFree(d_sp_va);  cudaFree(d_sp_rm);
    cudaFree(d_sp_t0);   cudaFree(d_sp_hp);  cudaFree(d_sp_cnt);
    cudaFree(d_traj);    cudaFree(d_next_sec);
    free(h_next_sec);

    /* release contiguous copies */
    Py_XDECREF(pos_arr); Py_XDECREF(mom_arr); Py_XDECREF(mass_arr);
    Py_XDECREF(charge_arr); Py_XDECREF(itime_arr); Py_XDECREF(B_arr);
    Py_XDECREF(z0_arr); Py_XDECREF(Bsig_arr); Py_XDECREF(layers_arr);
    Py_XDECREF(width_arr); Py_XDECREF(heights_arr); Py_XDECREF(angles_arr);
    Py_XDECREF(traj_arr);

    return ret;
  }

cleanup_arrs:
  Py_XDECREF(pos_arr); Py_XDECREF(mom_arr); Py_XDECREF(mass_arr);
  Py_XDECREF(charge_arr); Py_XDECREF(itime_arr); Py_XDECREF(B_arr);
  Py_XDECREF(z0_arr); Py_XDECREF(Bsig_arr); Py_XDECREF(layers_arr);
  Py_XDECREF(width_arr); Py_XDECREF(heights_arr); Py_XDECREF(angles_arr);
  Py_XDECREF(traj_arr);
  return NULL;
}

/* ── module definition ───────────────────────────────────────── */

static PyMethodDef methods[] = {
  {"solve", solve, METH_VARARGS,
   "CUDA version – same interface as straw_detector.solve()."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT, "straw_detector_cuda", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_straw_detector_cuda(void) {
  PyObject *m = PyModule_Create(&module_def);
  if (!m) return NULL;
  import_array();
  return m;
}
