#include "pythonWrapper/Loss.hpp"
#include "network/Loss.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void AddLoss(py::module_& m)
{
	py::class_<Loss>(m,"Loss");
	py::class_<MSE,Loss>(m,"MSE")
		.def(py::init<>());
	py::class_<CrossEntropy,Loss>(m,"CrossEntropy")
		.def(py::init<>());
}
