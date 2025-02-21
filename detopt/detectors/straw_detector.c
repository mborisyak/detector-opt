#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#define SPACE_DIM 3

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

static PyObject * straw_solve(PyObject *self, PyObject *args) {
  PyObject *dt_ = NULL;
  PyObject *J_ = NULL;
  PyObject *K_ = NULL;
  PyObject *L_ = NULL;
  PyObject *output_ = NULL;

  PyObject *initial_positions_ = NULL;
  PyObject *initial_velocities_ = NULL;
  PyObject *charges_ = NULL;
  PyObject *masses_ = NULL;

  if (!PyArg_UnpackTuple(
    args, "straw_solve", 9, 9,
    &initial_positions_,
    &initial_velocities_,
    &charges_,
    &masses_,
    &J_, &K_, &L_,
    &dt_, &output_
  )) {
    return NULL;
  }

  if (!PyArray_Check(output_)) {
    PyErr_SetString(
      PyExc_TypeError,
      "The output buffer must be an float64 array."
    );
    return NULL;
  }

  const PyArrayObject * output = (PyArrayObject *) output_;

  if (!(
    PyArray_TYPE(output) == NPY_FLOAT64 &&
    PyArray_NDIM(output) == 3 &&
    PyArray_DIM(output, 2) == SPACE_DIM
  )) {
    PyErr_SetString(
      PyExc_TypeError,
      "The output buffer must be a (n, n_t, 3) float64 array."
    );
    return NULL;
  }
  const npy_intp batch_size = PyArray_DIM(output, 0);
  const npy_intp steps = PyArray_DIM(output, 1);

  const PyArrayObject * initial_positions = check_vector_array(initial_positions_, batch_size);
  if (initial_positions == NULL) {
    PyErr_SetString(
      PyExc_TypeError,
      "Invalid value for initial positions provided. Must be a (n, 3) float64 array."
    );
    return NULL;
  }

  const PyArrayObject * initial_velocities = check_vector_array(initial_velocities_, batch_size);
  if (initial_velocities == NULL) {
    PyErr_SetString(
      PyExc_TypeError,
      "Invalid value for initial velocities provided. Must be a (n, 3) float64 array."
    );
    return NULL;
  }

  const PyArrayObject * masses = check_scalar_array(masses_, batch_size);
  if (masses == NULL) {
    PyErr_SetString(
      PyExc_TypeError,
      "Invalid value for masses provided. Must be a (n, ) float64 array."
    );
    return NULL;
  }

  const PyArrayObject * charges = check_scalar_array(charges_, batch_size);
  if (charges == NULL) {
    PyErr_SetString(
      PyExc_TypeError,
      "Invalid value for charges provided. Must be a (n, ) float64 array."
    );
    return NULL;
  }

  if (!PyFloat_Check(J_)) {
    PyErr_SetString(PyExc_TypeError, "J must be a double");
    return NULL;
  }
  if (!PyFloat_Check(K_)) {
      PyErr_SetString(PyExc_TypeError, "K must be a double");
      return NULL;
  }
  if (!PyFloat_Check(L_)) {
      PyErr_SetString(PyExc_TypeError, "L must be a double");
      return NULL;
  }

  const npy_double J = PyFloat_AsDouble(J_);
  const npy_double K = PyFloat_AsDouble(K_);
  const npy_double L = PyFloat_AsDouble(L_);

  if (!PyFloat_Check(dt_)) {
      PyErr_SetString(PyExc_TypeError, "dt must be a double");
      return NULL;
  }
  const npy_double dt = PyFloat_AsDouble(dt_);

  const npy_double * initial_positions_data = PyArray_DATA(initial_positions);
  const npy_double * initial_velocities_data = PyArray_DATA(initial_velocities);
  const npy_double * charges_data = PyArray_DATA(charges);
  const npy_double * masses_data = PyArray_DATA(masses);
  npy_double * output_data = PyArray_DATA(output);

  for (int i = 0; i < batch_size; ++i) {
    const npy_double c = 0.25 * dt * charges_data[i] / masses_data[i];
    npy_double x = initial_positions_data[i * SPACE_DIM];
    npy_double y = initial_positions_data[i * SPACE_DIM + 1];
    npy_double z = initial_positions_data[i * SPACE_DIM + 2];

    npy_double vx = initial_velocities_data[i * SPACE_DIM];
    npy_double vy = initial_velocities_data[i * SPACE_DIM + 1];
    npy_double vz = initial_velocities_data[i * SPACE_DIM + 2];

    for (int j = 0; j < steps; ++j) {
      const npy_double Bx = exp(-(x / L) * (x / L)) * (J * x + K * y);
      const npy_double By = -exp(-(x / L) * (x / L)) * (K * x - J * y);

      const npy_double tx = c * Bx;
      const npy_double ty = c * By;

      const npy_double t_norm_sqr = tx * tx + ty * ty;

      // placeholder
      output_data[i + j] = 1.0;
    }
  }

  return PyLong_FromLong(0);
}

static PyMethodDef StrawDetectorMethods[] = {
    {"solve",  straw_solve, METH_VARARGS, "Solve equations of motion."},
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