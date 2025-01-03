/*

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>

#include "datasetsBehaviour/DataLoader.h"

namespace py = pybind11;


DataLoader::DataLoader(py::array_t<float> input, py::array_t<float> output)
{
	if(input.shape()[0] != output.shape()[0])
		throw std::invalid_argument("DataLoader::DataLoader(py::array_t<double>,py::array_t<double) : The input length must be equal to the output length");

	size_t dataLength = input.shape()[0]; // Rows of input
	
	data = new MAT**[dataLength];
	

	//Do some barbary here but will come to later as we say TODO
	float* ptr_in = (float*)static_cast<const float*>(input.data());
	float* ptr_out = (float*)static_cast<const float*>(output.data());
	size_t in_mat_size = input.shape()[1] * input.shape()[2];	
	size_t out_mat_size = output.shape()[1] * output.shape()[2];
	for(size_t i = 0; i < dataLength; i++)
	{
		data[i] = new MAT*[2];
		data[i][0] = new Matrix(input.shape()[1],input.shape()[2],1,ptr_in);	
		data[i][1] = new Matrix(output.shape()[1],output.shape()[2],1,ptr_out);
		ptr_in += in_mat_size;	
		ptr_out += out_mat_size;
	}	

}

void AddDataLoader(py::module_& m)
{
	py::class_<DataLoader>(m,"Dataloader")
		.def(py::init<py::array_t<float>,py::array_t<float>>())
		.def("getSize",&DataLoader::GetSize);
		
}




*/