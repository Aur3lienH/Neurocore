from src.network.Activation import Activation
from src.network.Loss import Loss
from src.network.Layers import Layer

import subprocess
import threading
import importlib

import subprocess
import os
import sys


def RunCommand(command):
    # Set environment variables for color output
    my_env = os.environ.copy()
    my_env['PYTHONUNBUFFERED'] = '1'

    # Start process with direct output streaming
    process = subprocess.Popen(
        command,
        shell=True,
        env=my_env,
        # These are the key settings to preserve color
        stdout=None,
        stderr=None
    )
    
    # Wait for process to complete
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
    
    return return_code

def ImportLib(libpath,module_name):

    
    # Add the network directory to Python's path
    if "./build" not in sys.path:
        sys.path.append("./build")
    
    # Reload the module if it was previously imported
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Import the module and create the C++ network
    deep_learning_py = importlib.import_module(module_name)
    return deep_learning_py


class Network:
    def __init__(self):
        self.layers = []
        self.cpp_network = None
    
    def AddLayer(self, layer: Layer):
        self.layers.append(layer)

    def CompileCpp(self):
        # Get the absolute path to the project directory
        # This will be /home/aurelien/Projects/DeepLearning/python_lib
        project_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(project_dir)  # Go up one level from src/network
        project_dir = os.path.dirname(project_dir)  # Go up one more level to reach python_lib

        if not os.path.exists("build"):
            os.makedirs("build")

        
        # Save current directory
        original_dir = os.getcwd()
        
        try:
            # Change to project directory
            os.chdir(project_dir)
            
            # Run build commands
            RunCommand('cmake build -S build')
            RunCommand('make')
            
            deep_learning_py = ImportLib('./deep_learning_py*.so','deep_learning_py')

            self.cpp_network = deep_learning_py.Network()
            
        finally:
            # Always return to the original directory
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
        string += '    "Single input FeedForward")\n'
        string += '\t.def("BackPropagate", &NETWORK::BackPropagate)\n'
        string += '\t.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*)>(&NETWORK::Learn))\n'
        string += '//\t.def("Learn", static_cast<void (NETWORK::*)(int, double, DataLoader<NETWORK>*, int, int)>(&NETWORK::Learn))\n'
        string += '//\t.def("Process", &NETWORK::Process)\n'
        string += '//\t.def("ClearDelta", &NETWORK::ClearDelta)\n'
        string += '\t.def("Print", &NETWORK::PrintNetwork)\n'
        string += '\t.def("Compile", &NETWORK::Compile);\n'
        string += '}'
        file = open('build/network.cpp','w')
        file.write(string)
        file.close()
        self.CompileCpp()
        

    
    def FeedForward(self, input_data):
        if self.cpp_network is None:
            raise RuntimeError("Network not compiled. Call Compile() first.")
        return self.cpp_network.FeedForward(input_data)
    
    def Print(self):
        self.cpp_network.Print()
            
