#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define SPACE_DIM 3

#define SLOW_VZ 1.0e-3

inline npy_double square(npy_double x) {
  return x * x;
}

inline npy_double dabs(npy_double x) {
  return x > 0.0 ? x : -x;
}

const PyArrayObject * check_vector_array(const PyObject * object, int batch) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject * array = (PyArrayObject *) object;

  if (
    PyArray_TYPE(array) == NPY_FLOAT64 &&
    PyArray_NDIM(array) == 2 &&
    PyArray_DIM(array, 0) == batch &&
    PyArray_DIM(array, 1) == SPACE_DIM
  ) {
    return array;
  } else {
    return NULL;
  }
}

const PyArrayObject * check_scalar_array(const PyObject * object, int batch) {
  if (!PyArray_Check(object)) {
    return NULL;
  }
  const PyArrayObject * array = (PyArrayObject *) object;

  if (
    PyArray_Check(array) &&
    PyArray_TYPE(array) == NPY_FLOAT64 &&
    PyArray_NDIM(array) == 1 &&
    PyArray_DIM(array, 0) == batch
  ) {
    return array;
  } else {
    return NULL;
  }
}

// see documentation for the python method
static PyObject * straw_solve(PyObject *self, PyObject *args) {
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

  PyObject *py_trajectories = NULL;
  PyObject *py_response = NULL;

  if (!PyArg_UnpackTuple(
    args, "straw_solve", 13, 13,
    &py_initial_positions, &py_initial_velocities, &py_masses, &py_charges,
    &py_B, &py_L,
    &py_dt,
    &py_layers, &py_width, &py_heights, &py_angles,
    &py_trajectories, &py_response
  )) {
    return NULL;
  }

  if (!PyArray_Check(py_trajectories)) {
    PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be an float64 array.");
    return NULL;
  }

  if (!PyArray_Check(py_response)) {
    PyErr_SetString(PyExc_TypeError, "The response buffer must be an float64 array.");
    return NULL;
  }

  const PyArrayObject * trajectories_array = (PyArrayObject *) py_trajectories;
  const PyArrayObject * response_array = (PyArrayObject *) py_response;

  if (!(
    PyArray_TYPE(trajectories_array) == NPY_FLOAT64 &&
    PyArray_NDIM(trajectories_array) == 3 &&
    PyArray_DIM(trajectories_array, 2) == SPACE_DIM
  )) {
    PyErr_SetString(PyExc_TypeError, "The trajectories buffer must be a (n, n_t, 3) float64 array.");
    return NULL;
  }
  const npy_intp n_batch = PyArray_DIM(trajectories_array, 0);
  const npy_intp n_steps = PyArray_DIM(trajectories_array, 1);

  if (!(
    PyArray_TYPE(response_array) == NPY_FLOAT64 &&
    PyArray_NDIM(response_array) == 3 &&
    PyArray_DIM(response_array, 0) == n_batch
  )) {
    PyErr_SetString(PyExc_TypeError, "The response buffer must be a (n, n_layers, n_straws) float64 array.");
    return NULL;
  }
  const npy_intp n_layers = PyArray_DIM(response_array, 1);
  const npy_intp n_straws = PyArray_DIM(response_array, 2);

  const PyArrayObject * initial_positions_array = check_vector_array(py_initial_positions, n_batch);
  if (initial_positions_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "initial_positions must be a (n, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject * initial_velocities_array = check_vector_array(py_initial_velocities, n_batch);
  if (initial_velocities_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for initial velocities provided. Must be a (n, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject * masses_array = check_scalar_array(py_masses, n_batch);
  if (masses_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for masses provided. Must be a (n, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * charges_array = check_scalar_array(py_charges, n_batch);
  if (charges_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for charges provided. Must be a (n, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * layers_array = check_scalar_array(py_layers, n_layers);
  if (layers_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for layers' z-positions provided. Must be a (n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * width_array = check_scalar_array(py_width, n_layers);
  if (width_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for widths provided. Must be a (n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * angles_array = check_scalar_array(py_angles, n_layers);
  if (angles_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for angles provided. Must be a (n_layers, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * heights_array = check_scalar_array(py_heights, n_layers);
  if (heights_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for heights provided. Must be a (n_layers, ) float64 array.");
    return NULL;
  }

  if (!PyFloat_Check(py_B)) {
    PyErr_SetString(PyExc_TypeError, "B must be a double");
    return NULL;
  }
  if (!PyFloat_Check(py_L)) {
      PyErr_SetString(PyExc_TypeError, "L must be a double");
      return NULL;
  }
  if (!PyFloat_Check(py_dt)) {
      PyErr_SetString(PyExc_TypeError, "dt must be a double");
      return NULL;
  }

  const npy_double B = PyFloat_AsDouble(py_B);
  const npy_double L = PyFloat_AsDouble(py_L);
  const npy_double dt = PyFloat_AsDouble(py_dt);

  const npy_double * initial_positions = PyArray_DATA(initial_positions_array);
  const npy_double * initial_velocities = PyArray_DATA(initial_velocities_array);
  const npy_double * charges = PyArray_DATA(charges_array);
  const npy_double * masses = PyArray_DATA(masses_array);

  const npy_double * layers = PyArray_DATA(layers_array);
  const npy_double * widths = PyArray_DATA(width_array);
  const npy_double * angles = PyArray_DATA(angles_array);
  const npy_double * heights = PyArray_DATA(heights_array);

  npy_double * response = PyArray_DATA(response_array);
  npy_double * trajectories = PyArray_DATA(trajectories_array);

  for (int i = 0; i < n_batch; ++i) {
    npy_double x = initial_positions[i * SPACE_DIM];
    npy_double y = initial_positions[i * SPACE_DIM + 1];
    npy_double z = initial_positions[i * SPACE_DIM + 2];

    npy_double vx = initial_velocities[i * SPACE_DIM];
    npy_double vy = initial_velocities[i * SPACE_DIM + 1];
    npy_double vz = initial_velocities[i * SPACE_DIM + 2];

    npy_double v_sqr = vx * vx + vy * vy + vz * vz;

    const npy_double gamma = 1.0 / sqrt(1 - v_sqr);

    const npy_double charge = charges[i];
    const npy_double mass = masses[i];

    const npy_double c = 0.5 * dt * charge / mass / gamma;

    // motion through a purely magnetic field preserves |p|
    // npy_double px = vx * gamma * mass;
    // npy_double py = vy * gamma * mass;
    // npy_double pz = vz * gamma * mass;

    for (int j = 0; j < n_steps; ++j) {
      const npy_double Bx = B * exp(-square(z / L));
      const npy_double tx = c * Bx;
      const npy_double t_norm_sqr = tx * tx;

      // rotated speed
      // const npy_double vx_m = vx + vy * tz - vz * ty;
      // const npy_double vy_m = vy + vz * tx - vx * tz;
      // const npy_double vz_m = vz + vx * ty - vy * tx;
      // const npy_double vx_m = vx;
      const npy_double vy_m = vy + vz * tx;
      const npy_double vz_m = vz - vy * tx;

      const npy_double sx = 2 * tx / (1 + t_norm_sqr);

      vy = vy_m + vz_m * sx;
      // const npy_double vy = vy_m;
      vz = vz_m - vy_m * sx;

//      const npy_double cx = ay * bz - az * by;
//      const npy_double cy = az * bx - ax * bz;
//      const npy_double cz = ax * by - ay * bx;

      const npy_double dx = dt * vx;
      const npy_double dy = dt * vy;
      const npy_double dz = dt * vz;

      const npy_double x_ = x + dx;
      const npy_double y_ = y + dy;
      const npy_double z_ = z + dz;

      for (int k = 0; k < n_layers; ++k) {
        const npy_double layer = layers[k];
        const npy_double height = heights[k];
        const npy_double r = height / n_straws;

        if ((z < layer && z_ < layer) || (z > layer && z_ > layer)) {
          continue;
        }

        npy_double xi, yi;

        if (dabs(vz) < SLOW_VZ) {
          xi = (x + x_) / 2;
          yi = (y + y_) / 2;
        } else {
          xi = x + (layer - z) * vx / vz;
          yi = y + (layer - z) * vy / vz;
        }

        const npy_double angle = angles[k];
        const npy_double width = widths[k];

        const npy_double nx = cos(angle);
        const npy_double ny = sin(angle);

        const npy_double hx = nx * xi + ny * yi;
        const npy_double hy = -ny * xi + nx * yi;

        if (hx > width || hx < -width) {
          continue;
        }

        if (hy > height || hy < -height) {
          continue;
        }

        const npy_int straw_i = (npy_int) floor(0.5 * (hy + height) / r);

        if (straw_i >= 0 && straw_i < n_straws) {
          response[i * n_layers * n_straws + k * n_straws + straw_i] += 1.0;
        } else {
          printf(
            "Warning: invalid straw %d: y'=%lf (x=%lf, y=%lf, theta=%lf), H=%lf, r=%lf\n",
            straw_i, hy, xi, yi, angle, height, r
          );
        }
      }

      x = x_;
      y = y_;
      z = z_;

      // placeholder
      trajectories[i * n_steps * SPACE_DIM + j * SPACE_DIM] = x;
      trajectories[i * n_steps * SPACE_DIM + j * SPACE_DIM + 1] = y;
      trajectories[i * n_steps * SPACE_DIM + j * SPACE_DIM + 2] = z;
    }
  }

  return PyLong_FromLong(0);
}

static PyMethodDef StrawDetectorMethods[] = {
    {"solve",  straw_solve, METH_VARARGS, "Solve equations of motion and computes detector's response."},
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