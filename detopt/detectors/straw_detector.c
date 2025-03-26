#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define SPACE_DIM 3

inline npy_double square(npy_double x) {
  return x * x;
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

  PyObject *py_output = NULL;

  if (!PyArg_UnpackTuple(
    args, "straw_solve", 8, 8,
    &py_initial_positions,
    &py_initial_velocities,
    &py_masses, &py_charges,
    &py_B, &py_L,
    &py_dt, &py_output
  )) {
    return NULL;
  }

  if (!PyArray_Check(py_output)) {
    PyErr_SetString(PyExc_TypeError, "The output buffer must be an float64 array.");
    return NULL;
  }

  const PyArrayObject * output_array = (PyArrayObject *) py_output;

  if (!(
    PyArray_TYPE(output_array) == NPY_FLOAT64 &&
    PyArray_NDIM(output_array) == 3 &&
    PyArray_DIM(output_array, 2) == SPACE_DIM
  )) {
    PyErr_SetString(PyExc_TypeError, "The output buffer must be a (n, n_t, 3) float64 array.");
    return NULL;
  }
  const npy_intp batch_size = PyArray_DIM(output_array, 0);
  const npy_intp n_steps = PyArray_DIM(output_array, 1);

  const PyArrayObject * initial_positions_array = check_vector_array(py_initial_positions, batch_size);
  if (initial_positions_array == NULL) {
    PyErr_SetString(
      PyExc_TypeError,
      "Invalid value for initial positions provided. Must be a (n, 3) float64 array."
    );
    return NULL;
  }

  const PyArrayObject * initial_velocities_array = check_vector_array(py_initial_velocities, batch_size);
  if (initial_velocities_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for initial velocities provided. Must be a (n, 3) float64 array.");
    return NULL;
  }

  const PyArrayObject * masses_array = check_scalar_array(py_masses, batch_size);
  if (masses_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for masses provided. Must be a (n, ) float64 array.");
    return NULL;
  }

  const PyArrayObject * charges_array = check_scalar_array(py_charges, batch_size);
  if (charges_array == NULL) {
    PyErr_SetString(PyExc_TypeError, "Invalid value for charges provided. Must be a (n, ) float64 array.");
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
  npy_double * output = PyArray_DATA(output_array);

  for (int i = 0; i < batch_size; ++i) {
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
      const npy_double By = B * exp(-square(z / L));
      const npy_double ty = c * By;
      const npy_double t_norm_sqr = ty * ty;

      // rotated speed
      const npy_double vx_m = vx - vz * ty;
      // const npy_double vy_m = vy;
      const npy_double vz_m = vz + vx * ty;

      const npy_double sy = 2 * ty / (1 + t_norm_sqr);

      vx = vx_m - vz_m * sy;
      // const npy_double vy = vy_m;
      vz = vz_m + vx_m * sy;

//      const npy_double cx = ay * bz - az * by;
//      const npy_double cy = az * bx - ax * bz;
//      const npy_double cz = ax * by - ay * bx;

      x = x + dt * vx;
      y = y + dt * vy;
      z = z + dt * vz;

      // placeholder
      output[i * n_steps * SPACE_DIM + j * SPACE_DIM] = x;
      output[i * n_steps * SPACE_DIM + j * SPACE_DIM + 1] = y;
      output[i * n_steps * SPACE_DIM + j * SPACE_DIM + 2] = z;
    }
  }

  return PyLong_FromLong(0);
}

// see documentation for the python method
static PyObject * straw_detect(PyObject *self, PyObject *args) {
  // (n, n_t, 3)
  PyObject *py_trajectories= NULL;
  // (n_l, )
  PyObject *py_layers = NULL;
  // (n, n_l, n_s)
  PyObject *py_response = NULL;

  PyObject *py_size_x = NULL;
  PyObject *py_size_y = NULL;

  if (!PyArg_UnpackTuple(
    args, "straw_detect", 5, 5,
    &py_trajectories,
    &py_layers,
    &py_response,
    &py_size_x, &py_size_y
  )) {
    return NULL;
  }

  if (!PyArray_Check(py_trajectories)) {
    PyErr_SetString(PyExc_TypeError, "trajectories must be an float64 array.");
    return NULL;
  }
  const PyArrayObject * trajectories_array = (PyArrayObject *) py_trajectories;
  if (!(
    PyArray_TYPE(trajectories_array) == NPY_FLOAT64 &&
    PyArray_NDIM(trajectories_array) == 3 &&
    PyArray_DIM(trajectories_array, 2) == SPACE_DIM
  )) {
    PyErr_SetString(PyExc_TypeError, "trajectories must be a (n, n_t, 3) float64 array.");
    return NULL;
  }
  const npy_intp batch_size = PyArray_DIM(trajectories_array, 0);
  const npy_intp n_steps = PyArray_DIM(trajectories_array, 1);

  if (!PyArray_Check(py_layers)) {
    PyErr_SetString(PyExc_TypeError, "layers must be an float64 array.");
    return NULL;
  }
  const PyArrayObject * layers_array = (PyArrayObject *) py_layers;
  if (!(
    PyArray_TYPE(layers_array) == NPY_FLOAT64 &&
    PyArray_NDIM(layers_array) == 1
  )) {
    PyErr_SetString(PyExc_TypeError, "layers must be a (n_l, ) float64 array.");
    return NULL;
  }
  const npy_intp n_layers = PyArray_DIM(layers_array, 0);

  if (!PyFloat_Check(py_size_x)) {
    PyErr_SetString(PyExc_TypeError, "size_x must be a double");
    return NULL;
  }
  if (!PyFloat_Check(py_size_y)) {
      PyErr_SetString(PyExc_TypeError, "size_y must be a double");
      return NULL;
  }

  const npy_double size_x = PyFloat_AsDouble(py_size_x);
  const npy_double size_y = PyFloat_AsDouble(py_size_y);

  if (!PyArray_Check(py_response)) {
    PyErr_SetString(PyExc_TypeError, "response must be an float64 array.");
    return NULL;
  }
  const PyArrayObject * response_array = (PyArrayObject *) py_response;
  if (!(
    PyArray_TYPE(response_array) == NPY_FLOAT64 &&
    PyArray_NDIM(response_array) == 3 &&
    PyArray_DIM(response_array, 0) == batch_size &&
    PyArray_DIM(response_array, 1) ==  n_layers
  )) {
    PyErr_SetString(PyExc_TypeError, "response must be a (n, n_layers, n_straws) float64 array.");
    return NULL;
  }

  const npy_intp n_straws = PyArray_DIM(response_array, 2);

  const npy_double * const trajectories = PyArray_DATA(trajectories_array);
  const npy_double * const layers = PyArray_DATA(layers_array);
  npy_double * const response = PyArray_DATA(response_array);

  for (int i = 0; i < batch_size; ++i) {
    npy_double z_prev = trajectories[i * n_steps * SPACE_DIM + 2];

    for (int j = 1; j < n_steps; ++j) {
      npy_double z_curr = trajectories[i * n_steps * SPACE_DIM + j * SPACE_DIM + 2];

      // inefficient but universal algorithm, should be fine for a small number of layers
      for (int k = 0; k < n_layers; ++k) {
        npy_double z_layer = layers[k];
        if (z_prev < z_layer && z_layer < z_curr) {
          // hit
          printf("hit %d: %.3lf - %.3lf - %.3lf\n", i, z_prev, z_layer, z_curr);
          response[i * n_layers * n_straws + k * n_straws] = 1.0;
        }
      }

      z_prev = z_curr;
    }
  }

  return PyLong_FromLong(0);
}

static PyMethodDef StrawDetectorMethods[] = {
    {"solve",  straw_solve, METH_VARARGS, "Solve equations of motion."},
    {"detect",  straw_detect, METH_VARARGS, "computes straw detector response."},
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