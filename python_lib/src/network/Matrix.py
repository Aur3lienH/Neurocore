from src.network.Network import RunCommand, ImportLib
import os

class MatrixTypes:
    def __init__(self):
        self.matrixTypes = []

    def get_out_filename(self,rows,cols,dims):
        return f"build/matrix_{rows}x{cols}x{dims}.so"
    def get_module_name(self,rows,cols,dims):
        return f"matrix_{rows}x{cols}x{dims}"

    def add_lib(self,rows,cols,dims):
        if not os.path.exists("build"):
            os.makedirs("build")
        file = open(f'build/MAT_{rows}x{cols}x{dims}.cpp','w')
        file.write('#include "matrix/MatrixPy.hpp"\n')
        file.write('#include "matrix/Matrix.cuh"\n')
        file.write('#include <pybind11/pybind11.h>\n')
        file.write('#include <pybind11/stl.h>\n')
        file.write('namespace py = pybind11;\n')

        file.write(f'BIND_MATRIX({rows},{cols},{dims})')
        file.close()
        out_file_name = self.get_out_filename(rows,cols,dims)
        cmd = f"g++ -O3 -shared -std=c++20 -fPIC "\
          f"`python3 -m pybind11 --includes` "\
          f"-I ../dependencies/pybind11/include -I ../include "\
          f"./build/MAT_{rows}x{cols}x{dims}.cpp "\
          f"-o {out_file_name}"
        print(cmd)

        RunCommand(cmd)
        lib = ImportLib(f'{out_file_name}',f'{self.get_module_name(rows,cols,dims)}')
        self.matrixTypes.append(((rows,cols,dims),lib))
        return lib
    
    def get_lib(self,rows,cols,dims):
        for matrixType in self.matrixTypes:
            ((rows_lib,cols_lib,dims_lib),lib) = matrixType
            if rows_lib == rows and cols_lib == cols and dims_lib == dims:
                return lib
        return self.add_lib(rows,cols,dims)

        
matTypes = MatrixTypes()


class Matrix:

    def __init__(self, rows, cols, dims):
        self.rows = rows
        self.cols = cols
        self.dims = dims

        self.lib = matTypes.get_lib(rows,cols,dims).Matrix()

    def Print(self):
        self.lib.print()
