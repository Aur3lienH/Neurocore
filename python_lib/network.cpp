#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "network/Network.h"
#include "network/layers/FCL.cuh"
#include "network/activation/ReLU.h"
#include "network/layers/InputLayer.cuh"
#include "datasetsBehaviour/DataLoader.h"
#include "network/loss/MSE.cuh"
#include "network/loss/Loss.h"
#include <iostream>

namespace py = pybind11;

typedef Network<
	MSE<5,1,1>,
	InputLayer<LayerShape<1,1,1,1>>,
	FCL<ReLU<5,1,1,1>,LayerShape<1,1,1,1>,LayerShape<5,1,1,1>>
> NETWORK;

PYBIND11_MODULE(deep_learning_py, m) {
	py::class_<NETWORK>(m, "Network")
	.def(py::init<>())
.def("FeedForward",
   static_cast<const LMAT<typename NETWORK::OutputShape>* (NETWORK::*)(
        LMAT<typename NETWORK::InputShape>*)>(&NETWORK::FeedForward),
    "Single input FeedForward")
	.def("BackPropagate", &NETWORK::BackPropagate)
	.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*)>(&NETWORK::Learn))
//	.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*, int, int)>(&NETWORK::Learn))
//	.def("Process", &NETWORK::Process)
//	.def("ClearDelta", &NETWORK::ClearDelta)
	.def("Print", &NETWORK::PrintNetwork)
	.def("Compile", &NETWORK::Compile);
}