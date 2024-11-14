#include "pythonWrapper/Opti.hpp"
#include "network/Optimizers.cuh"
#include <pybind11/pybind11.h>

namespace py = pybind11;

void AddOpti(py::module_& m)
{
	py::enum_<Opti>(m,"Opti")
		.value("Constant", Opti::Constant)
		.value("Adam", Opti::Adam)
		.export_values();
}


