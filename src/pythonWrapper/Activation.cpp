#include "network/Activation.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

void AddActivations(py::module_& m)
{
	py::class_<Activation>(m,"Activaiton");
	py::class_<ReLU,Activation>(m,"ReLU")
		.def(py::init<>());
}
