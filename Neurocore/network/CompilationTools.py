import subprocess
import threading
import importlib
import os
import sys
import tempfile
from pathlib import Path
from ctypes.util import find_library
from Neurocore.network.Config import Config

build_dir = os.path.join(tempfile.mkdtemp(prefix='Neurocore'),'build')


def GetBuildDir():
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    return build_dir

def GetIncludeDir():
    neurocore_path = Path(__file__).parent.parent.absolute()
    include_dir = os.path.join(neurocore_path,'include')
    return include_dir

def GetPybindDir():
    neurocore_path = Path(__file__).parent.parent.absolute()
    pybind = os.path.join(neurocore_path,'dependencies','pybind11')
    return pybind

def GetTemplateDir():
    neurocore_path = Path(__file__).parent.parent.absolute()
    return os.path.join(neurocore_path,'templates')

def GetCudaDir():
    cuda_path = find_library('cudart')
    if cuda_path is None:
        raise ValueError("CUDA not found")
    return os.path.dirname(cuda_path)

def GetCompilationCommand(source_path, out_path):
    pybind_include = os.path.join(GetPybindDir(),'include')
    str = f"nvcc -O3 -shared -std=c++20 " \
           f"-I{GetCudaDir()}/include " \
           f"-Xcompiler -fPIC " \
           f"-x cu " \
           f" -arch=sm_75 " \
           f" `python3 -m pybind11 --includes`" \
           f" -I{pybind_include} -I{GetIncludeDir()} " \
           f" -L{GetCudaDir()}/lib64 -lcudart -lcublas -lcudnn " \
           f" `python3 -m pybind11 --includes` "
    if Config.USE_GPU:
        str += "-DUSE_GPU "
    str += f"{source_path} -o {out_path}"
    return str


def RunCommand(command):
    stdout = subprocess.DEVNULL
    if Config.VERBOSE:
        stdout = None
        print(f'Running : {command}')
    # Set environment variables for color output
    my_env = os.environ.copy()
    my_env['PYTHONUNBUFFERED'] = '1'
    # Start process with direct output streaming
    process = subprocess.Popen(
        command,
        shell=True,
        env=my_env,
        # These are the key settings to preserve color
        stdout=stdout,
        stderr=None
    )
    
    # Wait for process to complete
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
    
    return return_code

def ImportLib(module_path):
    # Get the module name from the .so file path
    module_name = Path(module_path).stem
    if module_name.startswith('lib'):  # Remove 'lib' prefix if present
        module_name = module_name[3:]
        
    # Remove from sys.modules if already loaded
    if module_name in sys.modules:
        del sys.modules[module_name]
        
    # Create module spec from file
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Could not create spec for {module_path}")
        
    # Create module from spec
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module
