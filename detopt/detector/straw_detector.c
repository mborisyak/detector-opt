#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define SPACE_DIM 3

#define SLOW_VZ 1.0e-3
#define SLOW 1.0e-6

inline npy_float square(npy_float x) {
  return x * x;
}

inline npy_float f32_abs(npy_float x) {
  return x > 0.0 ? x : -x;
}

inline npy_float f32_min(npy_float x, npy_float y) {
  return x > y ? y : x;
}

inline npy_float f32_max(npy_float x, npy_float y) {
  return x > y ? x : y;
}

inline npy_float imin(npy_int x, npy_int y) {
  return x > y ? y : x;
}

inline npy_float imax(npy_int x, npy_int y) {
  return x > y ? x : y;
}

const PyArrayObject * check_vector_array(const PyObject * object, int batch, int size) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject * array = (PyArrayObject *) object;

  if (
    // PyArray_IS_C_CONTIGUOUS(array) &&
    PyArray_TYPE(array) == NPY_FLOAT32 &&
    PyArray_NDIM(array) == 3 &&
    PyArray_DIM(array, 0) == batch &&
    PyArray_DIM(array, 1) == size &&
    PyArray_DIM(array, 2) == SPACE_DIM
  ) {
    return array;
  } else {
    return NULL;
  }
}

const PyArrayObject * check_scalar_array(const PyObject * object, int batch, int size) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject * array = (PyArrayObject *) object;

  if (
    // PyArray_IS_C_CONTIGUOUS(array) &&
    PyArray_Check(array) &&
    PyArray_TYPE(array) == NPY_FLOAT32 &&
    PyArray_NDIM(array) == 2 &&
    PyArray_DIM(array, 0) == batch &&
    PyArray_DIM(array, 1) == size
  ) {
    return array;
  } else {
    return NULL;
  }
}

const PyArrayObject * check_design_array(const PyObject * object, int batch) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject * array = (PyArrayObject *) object;

  if (
    //PyArray_IS_C_CONTIGUOUS(array) &&
    PyArray_Check(array) &&
    PyArray_TYPE(array) == NPY_FLOAT32 &&
    PyArray_NDIM(array) == 1 &&
    PyArray_DIM(array, 0) == batch
  ) {
    return array;
  } else {
    return NULL;
  }
}

// Helper function: check if point (x, y) is inside a convex quadrilateral (parallelogram)
int point_in_parallelogram(npy_float x, npy_float y, const npy_float corners[4][2]) {
    int i, j, sign = 0;
    for (i = 0; i < 4; ++i) {
        j = (i + 1) % 4;
        npy_float dx = corners[j][0] - corners[i][0];
        npy_float dy = corners[j][1] - corners[i][1];
        npy_float px = x - corners[i][0];
        npy_float py = y - corners[i][1];
        npy_float cross = dx * py - dy * px;
        if (cross == 0) continue;
        if (sign == 0) sign = (cross > 0) ? 1 : -1;
        else if ((cross > 0 && sign < 0) || (cross < 0 && sign > 0)) return 0;
    }
    return 1;
}

// see documentation for the python method
static PyObject * solve(PyObject *self, PyObject *args) {
  PyObject *py_dt= NULL;
  // field parameters
  PyObject *py_B = NULL;
  PyObject *py_L = NULL;

  // particle parameters
  PyObject *py_initial_positions = NULL;
  PyObject *py_initial_momenta = NULL;
  PyObject *py_masses = NULL;
  PyObject *py_charges = NULL;

  PyObject *py_layers = NULL;
  PyObject *py_width = NULL;
  PyObject *py_heights = NULL;
  PyObject *py_angles = NULL;

  PyObject *py_B_z0 = NULL;
  PyObject *py_B_sigma = NULL;

  PyObject *py_steps = NULL;

  PyObject *py_trajectories = NULL;
  PyObject *py_response = NULL;
  PyObject *py_edep = NULL;
  PyObject *py_r_mm = NULL;
  PyObject *py_t0 = NULL;
  PyObject *py_hit_pos = NULL;

  if (!PyArg_UnpackTuple(
       args, "straw_solve", 20, 20,
       &py_initial_positions, &py_initial_momenta, &py_masses, &py_charges,
       &py_B, &py_B_z0, &py_B_sigma,
       &py_steps, &py_dt,
       &py_layers, &py_width, &py_heights, &py_angles,
       &py_trajectories, &py_response, &py_edep, &py_r_mm, &py_t0, &py_hit_pos
   )) {
       return NULL;
   }

  if (!PyLong_Check(py_steps)) {
    PyErr_SetString(PyExc_TypeError, "steps must be an int");
    return NULL;
  }
  const long n_steps = PyLong_AsLong(py_steps);

  if (!PyArray_Check(py_response)) {
    PyErr_SetString(PyExc_TypeError, "The response buffer must be an float64 array.");
    return NULL;
  }

  const PyArrayObject * response_array = (PyArrayObject *) py_response;

  if (!(
    //PyArray_IS_C_CONTIGUOUS(response_array) &&
    PyArray_TYPE(response_array) == NPY_FLOAT32 &&
    PyArray_NDIM(response_array) == 4
  )) {
    PyErr_SetString(PyExc_TypeError, "The response buffer must be a (n, n_particles, n_layers, n_straws) float64 array.");
    return NULL;
  }
  const npy_intp n_batch = PyArray_DIM(response_array, 0);
  const npy_intp n_particles = PyArray_DIM(response_array, 1);
  const npy_intp n_layers = PyArray_DIM(response_array, 2);
  const npy_intp n_straws = PyArray_DIM(response_array, 3);

//  printf("batch: %ld, particles: %ld, layers: %ld straws: %ld\n", n_batch, n_particles, n_layers, n_straws);

  PyArrayObject * trajectories_array;

  if (!Py_IsNone(py_trajectories)) {
    if (!PyArray_Check(py_trajectories)) {
      PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be an float64 array.");
      return NULL;
    }
    trajectories_array = (PyArrayObject *) py_trajectories;

    if (!(
      //PyArray_IS_C_CONTIGUOUS(trajectories_array) &&
      PyArray_TYPE(trajectories_array) == NPY_FLOAT32 &&
      PyArray_NDIM(trajectories_array) == 4 &&
      PyArray_DIM(trajectories_array, 0) == n_batch &&
      PyArray_DIM(trajectories_array, 1) == n_particles &&
      PyArray_DIM(trajectories_array, 2) == n_steps &&
      PyArray_DIM(trajectories_array, 3) == SPACE_DIM
    )) {
      PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be a (n, n_particles, n_t, 3) float64 array.");
      return NULL;
    }
  } else {
    trajectories_array = NULL;
  }

  const PyArrayObject * initial_positions_array = check_vector_array(py_initial_positions, n_batch, n_particles);
  if (initial_positions_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "initial_positions must be a (n, n_particles, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject * initial_momenta_array = check_vector_array(py_initial_momenta, n_batch, n_particles);
  if (initial_momenta_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for initial momenta provided. Must be a (n, n_particles, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject * masses_array = check_scalar_array(py_masses, n_batch, n_particles);
  if (masses_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for masses provided. Must be a (n, n_particles) float64 array.");
    return NULL;
  }

  const PyArrayObject * charges_array = check_scalar_array(py_charges, n_batch, n_particles);
  if (charges_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for charges provided. Must be a (n, n_particles) float64 array.");
    return NULL;
  }

  const PyArrayObject * layers_array = check_scalar_array(py_layers, n_batch, n_layers);
  if (layers_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for layers' z-positions provided. Must be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * width_array = check_scalar_array(py_width, n_batch, n_layers);
  if (width_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for widths provided. Must be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * angles_array = check_scalar_array(py_angles, n_batch, n_layers);
  if (angles_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for angles provided. Must be a (n, n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * heights_array = check_scalar_array(py_heights, n_batch, n_layers);
  if (heights_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for heights provided. Must be a (n, n_layers, ) float64 array.");
    return NULL;
  }


  const PyArrayObject * B_array = check_design_array(py_B, n_batch);
  if (B_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for B provided. Must be a (n, ) float64 array.");
    return NULL;
  }
  const PyArrayObject * L_array = check_design_array(py_L, n_batch);
  if (L_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for L provided. Must be a (n, ) float64 array.");
    return NULL;
  }
  const PyArrayObject * B_z0_array = check_design_array(py_B_z0, n_batch);
  if (B_z0_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for z0 provided. Must be a (n, ) float64 array.");
    return NULL;
  }
  const PyArrayObject * B_sigma_array = check_design_array(py_B_sigma, n_batch);
  if (B_sigma_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for B_sigma provided. Must be a (n, ) float64 array.");
    return NULL;
  }

  if (!PyFloat_Check(py_dt)) {
      PyErr_SetString(PyExc_TypeError, "dt must be a double");
      return NULL;
  }
  const npy_float dt = PyFloat_AsDouble(py_dt);

  const npy_float * initial_positions = PyArray_DATA(initial_positions_array);
  const npy_float * initial_momenta = PyArray_DATA(initial_momenta_array);
  const npy_float * charges = PyArray_DATA(charges_array);
  const npy_float * masses = PyArray_DATA(masses_array);

  const npy_float * Bs = PyArray_DATA(B_array);
  const npy_float * Ls = PyArray_DATA(L_array);
  const npy_float * z0s = PyArray_DATA(B_z0_array);
  const npy_float * B_sigmas = PyArray_DATA(B_sigma_array);

  const npy_float * layers = PyArray_DATA(layers_array);
  const npy_float * widths = PyArray_DATA(width_array);
  const npy_float * angles = PyArray_DATA(angles_array);
  const npy_float * heights = PyArray_DATA(heights_array);

  npy_float * response = PyArray_DATA(response_array);
  npy_float * trajectories = (trajectories_array == NULL) ? NULL : PyArray_DATA(trajectories_array);

  // edep array: (n, n_particles, n_layers, n_straws)
  npy_float * edep = NULL;
  if (py_edep && py_edep != Py_None) {
    PyArrayObject * edep_array = (PyArrayObject *) py_edep;
    edep = PyArray_DATA(edep_array);
  }

  // r_mm array: (n, n_particles, n_layers, n_straws)
  npy_float * r_mm = NULL;
  if (py_r_mm && py_r_mm != Py_None) {
    PyArrayObject * r_mm_array = (PyArrayObject *) py_r_mm;
    r_mm = PyArray_DATA(r_mm_array);
  }
  // t0 array: (n, n_particles, n_layers, n_straws)
  npy_float * t0 = NULL;
  if (py_t0 && py_t0 != Py_None) {
    PyArrayObject * t0_array = (PyArrayObject *) py_t0;
    t0 = PyArray_DATA(t0_array);
  }
  // hit_pos array: (n, n_particles, n_layers, n_straws, 3)
  npy_float * hit_pos = NULL;
  if (py_hit_pos && py_hit_pos != Py_None) {
    PyArrayObject * hit_pos_array = (PyArrayObject *) py_hit_pos;
    hit_pos = PyArray_DATA(hit_pos_array);
  }


  npy_intp Bs0 = PyArray_STRIDE(B_array, 0) / sizeof(npy_float);
  npy_intp z0s0 = PyArray_STRIDE(z0_array, 0) / sizeof(npy_float);
  npy_intp B_sigma_s0 = PyArray_STRIDE(B_sigma_array, 0) / sizeof(npy_float);

  npy_intp ips0 = PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  npy_intp ips1 = PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);
  npy_intp ips2 = PyArray_STRIDE(initial_positions_array, 2) / sizeof(npy_float);

  npy_intp ivs0 = PyArray_STRIDE(initial_momenta_array, 0) / sizeof(npy_float);
  npy_intp ivs1 = PyArray_STRIDE(initial_momenta_array, 1) / sizeof(npy_float);
  npy_intp ivs2 = PyArray_STRIDE(initial_momenta_array, 2) / sizeof(npy_float);

  npy_intp chs0 = PyArray_STRIDE(charges_array, 0) / sizeof(npy_float);
  npy_intp chs1 = PyArray_STRIDE(charges_array, 1) / sizeof(npy_float);

  npy_intp ms0 = PyArray_STRIDE(masses_array, 0) / sizeof(npy_float);
  npy_intp ms1 = PyArray_STRIDE(masses_array, 1) / sizeof(npy_float);

  npy_intp ls0 = PyArray_STRIDE(layers_array, 0) / sizeof(npy_float);
  npy_intp ls1 = PyArray_STRIDE(layers_array, 1) / sizeof(npy_float);

  npy_intp hs0 = PyArray_STRIDE(heights_array, 0) / sizeof(npy_float);
  npy_intp hs1 = PyArray_STRIDE(heights_array, 1) / sizeof(npy_float);

  npy_intp ws0 = PyArray_STRIDE(width_array, 0) / sizeof(npy_float);
  npy_intp ws1 = PyArray_STRIDE(width_array, 1) / sizeof(npy_float);

  npy_intp as0 = PyArray_STRIDE(angles_array, 0) / sizeof(npy_float);
  npy_intp as1 = PyArray_STRIDE(angles_array, 1) / sizeof(npy_float);

  npy_intp rs0 = PyArray_STRIDE(response_array, 0) / sizeof(npy_float);
  npy_intp rs1 = PyArray_STRIDE(response_array, 1) / sizeof(npy_float);
  npy_intp rs2 = PyArray_STRIDE(response_array, 2) / sizeof(npy_float);
  npy_intp rs3 = PyArray_STRIDE(response_array, 3) / sizeof(npy_float);

  npy_intp trs0 = PyArray_STRIDE(trajectories_array, 0) / sizeof(npy_float);
  npy_intp trs1 = PyArray_STRIDE(trajectories_array, 1) / sizeof(npy_float);
  npy_intp trs2 = PyArray_STRIDE(trajectories_array, 2) / sizeof(npy_float);
  npy_intp trs3 = PyArray_STRIDE(trajectories_array, 3) / sizeof(npy_float);


  Py_BEGIN_ALLOW_THREADS

  for (int l = 0; l < n_batch; ++l) {
    // magnetic field parameters
    const npy_float B = Bs[l * Bs0];
    const npy_float z0 = z0s[l * z0s0];
    const npy_float B_sigma = B_sigmas[l * B_sigma_s0];

    for (int i = 0; i < n_particles; ++i) {
      npy_float x = initial_positions[l * ips0 + i * ips1];
      npy_float y = initial_positions[l * ips0 + i * ips1 + ips2];
      npy_float z = initial_positions[l * ips0 + i * ips1 + 2 * ips2];

      npy_float px = initial_momenta[l * ivs0 + i * ivs1];
      npy_float py = initial_momenta[l * ivs0 + i * ivs1 + ivs2];
      npy_float pz = initial_momenta[l * ivs0 + i * ivs1 + 2 * ivs2];

      const npy_float charge = charges[l * chs0 + i * chs1];
      const npy_float mass = masses[l * ms0 + i * ms1];

      npy_float p2 = px * px + py * py + pz * pz;
      const npy_float gamma = sqrtf(1.0f + p2 / (mass * mass));
      npy_float vx = px / (gamma * mass);
      npy_float vy = py / (gamma * mass);
      npy_float vz = pz / (gamma * mass);
      printf("gamma = %f mass = %f\n", gamma, mass);

      if (f32_abs(vx) < SLOW && f32_abs(vy) < SLOW && f32_abs(vz) < SLOW) {
        // ghost particle
        continue;
      }

      const npy_float c = 0.5 * dt * charge / mass / gamma;

      for (int j = 0; j < n_steps; ++j) {
        const npy_float Bx = B * exp(-square((z - z0) / B_sigma));
        const npy_float tx = c * Bx;
        const npy_float t_norm_sqr = tx * tx;

        // rotated speed
        // const npy_float vx_m = vx + vy * tz - vz * ty;
        // const npy_float vy_m = vy + vz * tx - vx * tz;
        // const npy_float vz_m = vz + vx * ty - vy * tx;
        // const npy_float vx_m = vx;
        const npy_float vy_m = vy + vz * tx;
        const npy_float vz_m = vz - vy * tx;

        const npy_float sx = 2 * tx / (1 + t_norm_sqr);

        vy = vy_m + vz_m * sx;
        vz = vz_m - vy_m * sx;

        const npy_float dx = dt * vx;
        const npy_float dy = dt * vy;
        const npy_float dz = dt * vz;

        const npy_float x_ = x + dx;
        const npy_float y_ = y + dy;
        const npy_float z_ = z + dz;

        for (int k = 0; k < n_layers; ++k) {
          const npy_float layer = layers[l * ls0 + k * ls1];
          const npy_float height = heights[l * hs0 + k * hs1];
          const npy_float width = widths[l * ws0 + k * ws1];
          const npy_float r = height / n_straws;
          const npy_float left = layer - r;
          const npy_float right = layer + r;

          // check for potential hit
          if ((z < left && z_ < left) || (z > right && z_ > right)) {
            continue;
          }

          const npy_float angle = angles[l * as0 + k * as1];

          const npy_float nx = cos(angle);
          const npy_float ny = sin(angle);

          const npy_float ry = -ny * x + nx * y;
          const npy_float rx = nx * x + ny * y;

          // Compute parallelogram corners in local (rx, ry) frame
          const npy_float skew = height * tanf(angle);
          npy_float corners[4][2] = {
            {-width - skew, -height},
            {-width + skew,  height},
            { width + skew,  height},
            { width - skew, -height}
          };

          // outside the parallelogram (frame)
          if (!point_in_parallelogram(rx, ry, corners)) {
            continue;
          }

          const npy_int straw_i = (npy_int) floor(0.5 * (ry + height) / r);
          const npy_float straw_y = (2 * straw_i + 1) * r - height;

          const npy_float sqr_distance_to_wire = square(z - layer) + square(ry - straw_y);

          if (sqr_distance_to_wire > r * r) {
            continue;
          }

          if (straw_i >= 0 && straw_i < n_straws) {
            response[l * rs0 + i * rs1 + k * rs2 + straw_i * rs3] += dt;
            // Store hit position (x, y, z) for first hit
            if (hit_pos) {
              npy_intp idx = l * rs0 + i * rs1 + k * rs2 + straw_i * rs3;
              if (response[idx] == dt) { // first hit for this straw
                hit_pos[idx * 3 + 0] = x;
                hit_pos[idx * 3 + 1] = y;
                hit_pos[idx * 3 + 2] = z;
              }
            }

           // Store time of first hit (t0, in ns)
           if (t0) {
             npy_intp idx = l * rs0 + i * rs1 + k * rs2 + straw_i * rs3;
             if (response[idx] == dt) { // first hit for this straw in this event/particle/layer
               t0[idx] = j * dt;
             }
           }

            // Store transverse distance to wire (r_mm)
            if (r_mm) {
              r_mm[l * rs0 + i * rs1 + k * rs2 + straw_i * rs3] = fabsf(ry - straw_y);
            }

            // --- Bethe-Bloch energy loss calculation ---
            // Constants for Argon gas (default)
            const npy_float K = 0.307075; // MeV*cm^2/g
            const npy_float Z = 18.0;     // Argon
            const npy_float A = 39.948;   // Argon
            const npy_float I_exc = 188e-6;   // MeV
            const npy_float rho = 1.66e-3; // g/cm^3
            const npy_float me = 0.511;   // MeV/c^2

            // Compute beta, gamma from momentum and mass
            npy_float p = sqrtf(px*px + py*py + pz*pz);
            npy_float beta = p / sqrtf(p*p + mass*mass);
            npy_float gamma_bethe = sqrtf(1.0f + (p*p) / (mass*mass));
            // Tmax for Bethe-Bloch
            npy_float Tmax = (2 * me * beta * beta * gamma_bethe * gamma_bethe) /
              (1 + 2 * gamma_bethe * me / 0.938 + (me / 0.938) * (me / 0.938));
            npy_float argument = (2 * me * beta * beta * gamma_bethe * gamma_bethe * Tmax) / (I_exc*I_exc);
            if (argument <= 0) argument = 1e-10;
            npy_float log_term = logf(argument);
            npy_float dEdx = K * (charge*charge) * Z / A / (beta*beta) * (0.5f * log_term - beta*beta) * rho; // MeV/cm

            // Path length in this step (cm)
            npy_float v = sqrtf(vx * vx + vy * vy + vz * vz);
            npy_float path_cm = v * dt * 10.0f; // dt in ns, v in mm/ns, convert mm to cm

            npy_float Edep = dEdx * path_cm; // MeV deposited in this step

            if (edep) {
              edep[l * rs0 + i * rs1 + k * rs2 + straw_i * rs3] += Edep;
            }
          } else {
            printf(
              "Warning: invalid straw %d: y'=%lf (x=%lf, y=%lf, theta=%lf), H=%lf, r=%lf\n",
              straw_i, ry, x, y, angle, height, r
            );
          }
        }

        x = x_;
        y = y_;
        z = z_;

        if (trajectories != NULL) {
          trajectories[l * trs0 + i * trs1 + j * trs2] = x;
          trajectories[l * trs0 + i * trs1 + j * trs2 + trs3] = y;
          trajectories[l * trs0 + i * trs1 + j * trs2 + 2 * trs3] = z;
        }
      }
    }
  }

//  printf("%d hits vs %d misses: %.3lf\n", hits, misses, ((double) misses) / (hits + misses));

  Py_END_ALLOW_THREADS

  return PyLong_FromLong(0);
}

static PyMethodDef StrawDetectorMethods[] = {
    {"solve",  solve, METH_VARARGS, "Solve equations of motion and computes detector's response."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef straw_detector_module = {
    PyModuleDef_HEAD_INIT,
    "straw_detector",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    StrawDetectorMethods
};

PyMODINIT_FUNC
PyInit_straw_detector(void)
{
    PyObject *m;
    m = PyModule_Create(&straw_detector_module);
    if (m == NULL) return NULL;

    import_array();  // Initialize the NumPy API

    return m;
}
