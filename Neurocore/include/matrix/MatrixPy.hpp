#include "matrix/Matrix.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>





#define BIND_MATRIX(ROWS, COLS, DIMS)\
PYBIND11_MODULE(matrix_##ROWS##x##COLS##x##DIMS, m) { \
    py::class_<MAT<ROWS,COLS,DIMS>>(m, "Matrix") \
        .def(py::init<>()) \
        .def(py::init<py::array_t<float>&>()) \
        .def("to_numpy", &MAT<ROWS,COLS,DIMS>::ToNumpy) \
        .def("get_rows", &MAT<ROWS,COLS,DIMS>::GetRows) \
        .def("get_cols", &MAT<ROWS,COLS,DIMS>::GetCols) \
        .def("print", &MAT<ROWS,COLS,DIMS>::Print) \
        .def("convert_to_array", &MAT<ROWS,COLS,DIMS>::ConvertToArray,py::return_value_policy::take_ownership); \
}
