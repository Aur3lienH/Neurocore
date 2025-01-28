from src.network.Activation import Activation
from src.network.Loss import Loss
from src.network.Layers import Layer
from src.network.Matrix import Matrix
from src.network.CompilationTools import RunCommand, ImportLib
import numpy as np
import numpy.typing as npt
from src.network.MatrixTypes import MatrixTypes
import os

from src.network.Matrix import matTypes, NumpyToMatrixArray


class Network:
    def __init__(self):
        self.layers = []
        self.cpp_network = None
    
    def AddLayer(self, layer: Layer):
        self.layers.append(layer)

    def CompileCpp(self):
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        build_dir = os.path.join(project_dir, "build")

        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        original_dir = os.getcwd()
        cache = os.path.join(build_dir, 'CMakeCache.txt')
        makefile = os.path.join(build_dir, 'Makefile')
        cmake_install = os.path.join(build_dir, 'cmake_install.cmake')
        if os.path.exists(cache):
            os.remove(cache)
        if os.path.exists(makefile):
            os.remove(makefile)
        if os.path.exists(cmake_install):
            os.remove(cmake_install)

        try:
            os.chdir(project_dir)
            RunCommand('cmake build -S build -B build -DPython3_FIND_STRATEGY=LOCATION')
            RunCommand('cd build && make')

            # Use absolute path for module import
            module_path = os.path.join(project_dir, "build", 'deep_learning_py*.so')
            print(module_path)
            deep_learning_py = ImportLib('deep_learning_py')

            self.cpp_network = deep_learning_py.Network()
            self.cpp_lib_core = deep_learning_py

        finally:
            os.chdir(original_dir)
    
    def Compile(self, loss: Loss):

        self.loss = loss
        string = ''
        string += '#include <pybind11/pybind11.h>\n'
        string += '#include <pybind11/stl.h>\n'
        string += '#include "network/Network.h"\n'
        string += '#include "network/layers/FCL.cuh"\n'
        string += '#include "network/activation/ReLU.h"\n'
        string += '#include "network/layers/InputLayer.cuh"\n'
        string += '#include "datasetsBehaviour/DataLoader.h"\n'
        string += '#include "network/loss/MSE.cuh"\n'
        string += '#include "network/loss/Loss.h"\n'
        string += '#include "datasetsBehaviour/DataLoader.h"\n'
        string += '#include <cstddef>\n'
        string += '#include <iostream>\n\n'
        string += 'namespace py = pybind11;\n\n'

        string += f'typedef Network<\n'
        string += f'\t{loss.get_code(self.layers[-1].layerShape)}'
        prev_shape = None
        for layer in self.layers:
            string += ',\n'
            string += '\t'
            string += layer.get_code(prev_shape)
            prev_shape = layer.get_layer_shape()
        string += f'\n> NETWORK;\n\n'

        string += 'PYBIND11_MODULE(deep_learning_py, m) {\n'
        string += '\tpy::class_<NETWORK>(m, "Network")\n'
        string += '\t.def(py::init<>())\n'
        string += '.def("FeedForward",\n' 
        string += '   static_cast<const LMAT<typename NETWORK::OutputShape>* (NETWORK::*)(\n'
        string += '        LMAT<typename NETWORK::InputShape>*)>(&NETWORK::FeedForward),\n'
        string += '    "Single input FeedForward",py::return_value_policy::reference)\n'
        string += '\t.def("BackPropagate", &NETWORK::BackPropagate)\n'
        string += '\t.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*)>(&NETWORK::Learn))\n'
        string += '//\t.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*, int, int)>(&NETWORK::Learn))\n'
        string += '//\t.def("Process", &NETWORK::Process)\n'
        string += '//\t.def("ClearDelta", &NETWORK::ClearDelta)\n'
        string += '\t.def("Print", &NETWORK::PrintNetwork)\n'
        string += '\t.def("Compile", &NETWORK::Compile);\n'
        string += '\tpy::class_<DataLoader<NETWORK>>(m, "DataLoader")\n'
        string += '\t.def(py::init<py::object,py::object,size_t>());\n'
        string += '}'
        file = open('build/network.cpp','w')
        file.write(string)
        file.close()
        self.CompileCpp()
        self.cpp_network.Compile()
        
        

    
    def FeedForward(self, input_data: npt.NDArray[np.float32]):
        mat_data = Matrix(numpyArray=input_data)
        if self.cpp_network is None:
            raise RuntimeError("Network not compiled. Call Compile() first.")
        out_res = self.cpp_network.FeedForward(mat_data.cpp_mat)
        res_shape = self.layers[-1].get_layer_shape()
        mat = Matrix(res_shape.x,res_shape.y,res_shape.z,out_res)
        mat.Print()
        return mat.cpp_mat.to_numpy()
    

    def Learn(self, input_data, output_desired, batch_size = 1, epochs = 1, learning_rate = 0.01):
        if(self.cpp_network is None):
            raise RuntimeError("Network not compile. Call Compile first.")
        input_shape = input_data.shape
        output_shape = output_desired.shape
        if input_shape[0] != output_shape[0]:
            raise ValueError("Input and output shapes do not match")
        input = NumpyToMatrixArray(input_data)
        output = NumpyToMatrixArray(output_desired)
        print(input_data)
        print(output_desired)
        print(input)
        print(output)
        self.cpp_data_loader = self.cpp_lib_core.DataLoader(input,output,input_shape[0])
        self.cpp_network.Learn(epochs,learning_rate,self.cpp_data_loader)
        
    
    def Print(self):
        self.cpp_network.Print()
            
