#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "network/Network.h"
#include "pythonWrapper/Activations.hpp"
#include "pythonWrapper/Layers.hpp"
#include "pythonWrapper/DataLoader.hpp"
#include "pythonWrapper/Loss.hpp"
#include "pythonWrapper/Opti.hpp"

namespace py = pybind11;


PYBIND11_MODULE(libdeep, m)
{
	py::class_<Network>(m,"Network")
		.def(py::init<>())
		.def("Print",&Network::PrintNetwork)
		.def("Learn",py::overload_cast<int,double,DataLoader*,int, int>(&Network::Learn))
		.def("FeedForward",py::overload_cast<MAT*>(&Network::FeedForward))
		.def("AddLayer",py::overload_cast<Layer*>(&Network::AddLayer))
		.def("Save",py::overload_cast<const std::string&>(&Network::Save))
		.def("Compile",py::overload_cast<Opti,Loss*>(&Network::Compile));
	AddLayers(m);
	AddActivations(m);
	AddDataLoader(m);
	AddLoss(m);
	AddOpti(m);
}
