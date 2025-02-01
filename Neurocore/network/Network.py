from Neurocore.network.Activation import Activation
from Neurocore.network.Loss import Loss
from Neurocore.network.Layers import Layer
from Neurocore.network.Matrix import Matrix
from Neurocore.network.CompilationTools import RunCommand, ImportLib, GetBuildDir, GetTemplateDir, GetPybindDir, GetIncludeDir
import numpy as np
import numpy.typing as npt
from Neurocore.network.MatrixTypes import MatrixTypes
import os
from Neurocore.network.Config import Config
from Neurocore.network.Matrix import matTypes, NumpyToMatrixArray, PreCompileMatrix


class Network:
    def __init__(self):
        self.layers = []
        self.cpp_network = None
    
    def AddLayer(self, layer: Layer):
        self.layers.append(layer)

    def CompileCpp(self):
        build_dir = GetBuildDir()



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
            cmake_list_path = os.path.join(GetTemplateDir(),'CMakeLists.txt')
            source_path = os.path.join(build_dir,'network.cpp')
            out_path = os.path.join(build_dir,'neurocore.so')
            pybind_include = os.path.join(GetPybindDir(),'include')
            cmd = f"g++ -O3 -shared -std=c++20 -fPIC -flto=auto "\
          f" `python3 -m pybind11 --includes`"\
          f" -I{pybind_include} -I{GetIncludeDir()} "\
          f"{source_path} "\
          f"-o {out_path}"
            #RunCommand(f'cp {cmake_list_path} {build_dir}')
            #RunCommand(f'cmake -DPYBIND11_ROOT={GetPybindDir()} -DINCLUDE_DIR={GetIncludeDir()} -DNETWORK_FILE={source_path} -S {build_dir} -B {build_dir} -DPython3_FIND_STRATEGY=LOCATION')
            #RunCommand(f'cd {build_dir} && make')
            RunCommand(cmd)

            # Use absolute path for module import
            module_path = os.path.join(build_dir, 'neurocore.so')
            deep_learning_py = ImportLib(module_path)

            self.cpp_network = deep_learning_py.Network()
            self.cpp_lib_core = deep_learning_py
            if Config.VERBOSE:
                print('Network: Compiled !')

        finally:
            os.chdir(original_dir)
    
    def Compile(self, loss: Loss):
        build_dir = GetBuildDir()

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

        string += 'PYBIND11_MODULE(neurocore, m) {\n'
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
        file = open(os.path.join(build_dir,'network.cpp'),'w')
        file.write(string)
        file.close()
        self.CompileCpp()
        first_layer = self.layers[0].get_layer_shape()
        last_layer = self.layers[-1].get_layer_shape()
        PreCompileMatrix(first_layer.x,first_layer.y,first_layer.z)
        PreCompileMatrix(last_layer.x,last_layer.y,last_layer.z)
        self.cpp_network.Compile()
        
        

    
    def FeedForward(self, input_data: npt.NDArray[np.float32]):
        mat_data = Matrix(numpyArray=input_data)
        if self.cpp_network is None:
            raise RuntimeError("Network not compiled. Call Compile() first.")
        out_res = self.cpp_network.FeedForward(mat_data.cpp_mat)
        res_shape = self.layers[-1].get_layer_shape()
        mat = Matrix(res_shape.x,res_shape.y,res_shape.z,out_res)
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
        self.cpp_data_loader = self.cpp_lib_core.DataLoader(input,output,input_shape[0])
        self.cpp_network.Learn(epochs,learning_rate,self.cpp_data_loader)
        
    
    def Print(self):
        self.cpp_network.Print()
            
