from Neurocore.network.CompilationTools import RunCommand, ImportLib, GetBuildDir, GetIncludeDir, GetPybindDir
import os
from Neurocore.network.Config import Config

class MatrixTypes:
    def __init__(self):
        self.matrixTypes = []

    def get_out_filename(self,rows,cols,dims):
        return f"matrix_{rows}x{cols}x{dims}.so"
    def get_module_name(self,rows,cols,dims):
        return f"matrix_{rows}x{cols}x{dims}"

    def add_lib(self,rows,cols,dims):
        if Config.VERBOSE:
            print(f'Compiling: Matrix with size {rows}x{cols}x{dims}')
        build_dir = GetBuildDir()
        filepath = os.path.join(build_dir,f'MAT_{rows}x{cols}x{dims}.cpp')
        file = open(filepath,'w')
        file.write('#include "matrix/MatrixPy.hpp"\n')
        file.write('#include "matrix/Matrix.cuh"\n')
        file.write('#include <pybind11/pybind11.h>\n')
        file.write('#include <pybind11/stl.h>\n')
        file.write('namespace py = pybind11;\n')

        file.write(f'BIND_MATRIX({rows},{cols},{dims})')
        file.close()
        out_file_path = os.path.join(build_dir,self.get_out_filename(rows,cols,dims))
        pybind_include = os.path.join(GetPybindDir(),'include')
        cmd = f"g++ -O3 -shared -std=c++20 -fPIC -flto=auto "\
          f" `python3 -m pybind11 --includes`"\
          f" -I{pybind_include} -I{GetIncludeDir()} "\
          f"{filepath} "\
          f"-o {out_file_path}"

        RunCommand(cmd)
        lib = ImportLib(out_file_path)
        self.matrixTypes.append(((rows,cols,dims),lib))
        return lib
    
    def compile_lib(self,rows,cols,dims):
        self.get_lib(rows,cols,dims)
    
    def get_lib(self,rows,cols,dims):
        for matrixType in self.matrixTypes:
            ((rows_lib,cols_lib,dims_lib),lib) = matrixType
            if rows_lib == rows and cols_lib == cols and dims_lib == dims:
                if Config.VERBOSE:
                    print(f'Retriving: Already compiled Matrix with size {rows}x{cols}x{dims}')
                return lib
        lib = self.add_lib(rows,cols,dims)
        return lib