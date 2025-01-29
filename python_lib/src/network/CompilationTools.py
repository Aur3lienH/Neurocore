import subprocess
import threading
import importlib
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

def ImportLib(module_name):


    # Add the network directory to Python's path
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build_dir = os.path.join(project_dir, "build")
    sys.path.append(build_dir)

    
    # Reload the module if it was previously imported
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Import the module and create the C++ network
    deep_learning_py = importlib.import_module(module_name)
    return deep_learning_py
