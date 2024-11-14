#include <pybind11/pybind11.h>

#include "network/layers/ConvLayer.cuh"
#include "network/layers/FCL.cuh"
#include "network/layers/AveragePooling.cuh"
#include "network/layers/Flatten.cuh"
#include "network/layers/InputLayer.cuh"
#include "network/layers/MaxPooling.cuh"
#include "network/layers/DropoutFCL.cuh"


namespace py = pybind11;

void AddLayers(py::module_& m)
{

py::class_<Layer>(m,"Layer");

py::class_<ConvLayer,Layer>(m,"ConvLayer")
	.def(py::init<LayerShape*,Activation*>());

py::class_<FCL,Layer>(m,"FullLayer")
	.def(py::init<int,Activation*>());

py::class_<InputLayer,Layer>(m,"InputLayer")
	.def(py::init<int>());
}

