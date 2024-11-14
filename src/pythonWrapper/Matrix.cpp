#include "matrix/Matrix.cuh"

#include <pybind11/numpy.h>

namespace py = pybind11;


Matrix::Matrix(py::array_t<float> input)
{	
	Matrix(input.shape()[0],input.shape()[1],input.shape()[2],(float*)(static_cast<const float*>(input.data())));
}
