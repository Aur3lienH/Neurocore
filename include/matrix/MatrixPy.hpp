#include "matrix/Matrix.cuh"
#include <pybind11/pybind11.h>
#include <iostream>


#define BIND_MATRIX(ROWS, COLS, DIMS)\
PYBIND11_MODULE(matrix_##ROWS##x##COLS##x##DIMS, m) { \
    py::class_<MAT<ROWS,COLS,DIMS>>(m, "Matrix") \
        .def(py::init<>()) \
        .def("get_rows", &MAT<ROWS,COLS,DIMS>::GetRows) \
        .def("get_cols", &MAT<ROWS,COLS,DIMS>::GetCols) \
        .def("print", &MAT<ROWS,COLS,DIMS>::Print); \
}
