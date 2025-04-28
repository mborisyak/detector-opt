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
  return x > y ? y : x;
}

inline npy_float imin(npy_int x, npy_int y) {
  return x > y ? y : x;
}

inline npy_float imax(npy_int x, npy_int y) {
  return x > y ? y : x;
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

// see documentation for the python method
static PyObject * solve(PyObject *self, PyObject *args) {
  PyObject *py_dt= NULL;
  // field parameters
  PyObject *py_B = NULL;
  PyObject *py_L = NULL;

  // particle parameters
  PyObject *py_initial_positions = NULL;
  PyObject *py_initial_velocities = NULL;
  PyObject *py_charges = NULL;
  PyObject *py_masses = NULL;

  PyObject *py_layers = NULL;
  PyObject *py_width = NULL;
  PyObject *py_heights = NULL;
  PyObject *py_angles = NULL;

  PyObject *py_steps = NULL;

  PyObject *py_trajectories = NULL;
  PyObject *py_response = NULL;

  if (!PyArg_UnpackTuple(
    args, "straw_solve", 14, 14,
    &py_initial_positions, &py_initial_velocities, &py_masses, &py_charges,
    &py_B, &py_L,
    &py_steps, &py_dt,
    &py_layers, &py_width, &py_heights, &py_angles,
    &py_trajectories, &py_response
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

  const PyArrayObject * initial_velocities_array = check_vector_array(py_initial_velocities, n_batch, n_particles);
  if (initial_velocities_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for initial velocities provided. Must be a (n, n_particles, 3) float64 array.");
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

  if (!PyFloat_Check(py_dt)) {
      PyErr_SetString(PyExc_TypeError, "dt must be a double");
      return NULL;
  }
  const npy_float dt = PyFloat_AsDouble(py_dt);

  const npy_float * initial_positions = PyArray_DATA(initial_positions_array);
  const npy_float * initial_velocities = PyArray_DATA(initial_velocities_array);
  const npy_float * charges = PyArray_DATA(charges_array);
  const npy_float * masses = PyArray_DATA(masses_array);

  const npy_float * Bs = PyArray_DATA(B_array);
  const npy_float * Ls = PyArray_DATA(L_array);

  const npy_float * layers = PyArray_DATA(layers_array);
  const npy_float * widths = PyArray_DATA(width_array);
  const npy_float * angles = PyArray_DATA(angles_array);
  const npy_float * heights = PyArray_DATA(heights_array);

  npy_float * response = PyArray_DATA(response_array);
  npy_float * trajectories = (trajectories_array == NULL) ? NULL : PyArray_DATA(trajectories_array);

  npy_intp Bs0 = PyArray_STRIDE(B_array, 0) / sizeof(npy_float);
  npy_intp Ls0 = PyArray_STRIDE(L_array, 0) / sizeof(npy_float);

  npy_intp ips0 = PyArray_STRIDE(initial_positions_array, 0) / sizeof(npy_float);
  npy_intp ips1 = PyArray_STRIDE(initial_positions_array, 1) / sizeof(npy_float);
  npy_intp ips2 = PyArray_STRIDE(initial_positions_array, 2) / sizeof(npy_float);

  npy_intp ivs0 = PyArray_STRIDE(initial_velocities_array, 0) / sizeof(npy_float);
  npy_intp ivs1 = PyArray_STRIDE(initial_velocities_array, 1) / sizeof(npy_float);
  npy_intp ivs2 = PyArray_STRIDE(initial_velocities_array, 2) / sizeof(npy_float);

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
    const npy_float B = Bs[l * Bs0];
    const npy_float L = Ls[l * Ls0];

    for (int i = 0; i < n_particles; ++i) {
      npy_float x = initial_positions[l * ips0 + i * ips1];
      npy_float y = initial_positions[l * ips0 + i * ips1 + ips2];
      npy_float z = initial_positions[l * ips0 + i * ips1 + 2 * ips2];

      npy_float vx = initial_velocities[l * ivs0 + i * ivs1];
      npy_float vy = initial_velocities[l * ivs0 + i * ivs1 + ivs2];
      npy_float vz = initial_velocities[l * ivs0 + i * ivs1 + 2 * ivs2];

      if (f32_abs(vx) < SLOW && f32_abs(vy) < SLOW && f32_abs(vz) < SLOW) {
        // ghost particle
        continue;
      }

      npy_float v_sqr = vx * vx + vy * vy + vz * vz;

      const npy_float gamma = 1.0 / sqrt(1 - v_sqr);

      const npy_float charge = charges[l * chs0 + i * chs1];
      const npy_float mass = masses[l * ms0 + i * ms1];

      const npy_float c = 0.5 * dt * charge / mass / gamma;

      // printf("event %d, particle %d: %.2lf %.2lf %.2lf (c=%.2e)\n", l, i, vx, vy, vz, c);

      // motion through a purely magnetic field preserves |p|
      // npy_float px = vx * gamma * mass;
      // npy_float py = vy * gamma * mass;
      // npy_float pz = vz * gamma * mass;

      for (int j = 0; j < n_steps; ++j) {
        const npy_float Bx = B * exp(-square(z / L));
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
        // const npy_float vy = vy_m;
        vz = vz_m - vy_m * sx;

  //      const npy_float cx = ay * bz - az * by;
  //      const npy_float cy = az * bx - ax * bz;
  //      const npy_float cz = ax * by - ay * bx;

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

          // outside the layer
          if (f32_abs(rx) > width || f32_abs(ry) > height) {
            continue;
          }

          const npy_int straw_i = (npy_int) floor(0.5 * (ry + height) / r);
          const npy_float straw_y = (2 * straw_i + 1) * r - height;

          const npy_float sqr_distance_to_wire = square(z - layer) + square(ry - straw_y);

          if (sqr_distance_to_wire > r * r) {
//            printf(
//              "Warning: missing a straw %d: z=%.3lf (%.3lf), y'=%.3lf (%.3lf), r=%.3lf, d=%.3lf\n",
//              straw_i, z, layer, ry, straw_y, r, sqrt(sqr_distance_to_wire)
//            );
            continue;
          }

          if (straw_i >= 0 && straw_i < n_straws) {
            response[l * rs0 + i * rs1 + k * rs2 + straw_i * rs3] += dt;
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