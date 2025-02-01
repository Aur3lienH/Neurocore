from Neurocore.network.MatrixTypes import MatrixTypes
import numpy as np
import numpy.typing as npt
from Neurocore.network.Config import Config
        
matTypes = MatrixTypes()

def NumpyToMatrixArray(numpyArray: npt.NDArray[np.float32]):
    if len(numpyArray.shape) == 3:
        lib = matTypes.get_lib(numpyArray.shape[1],numpyArray.shape[2],1)
    elif len(numpyArray.shape) == 2:
        lib = matTypes.get_lib(numpyArray.shape[1],1,1)
    else:
        raise ValueError("Invalid numpy array shape")
    return lib.Matrix.convert_to_array(numpyArray)

def PreCompileMatrix(rows, cols, dims):
    matTypes.get_lib(rows,cols,dims)


class Matrix:

    def __init__(self, rows = 1, cols = 1, dims = 1, cpp_mat = None, numpyArray: npt.NDArray[np.float32] = None):
        if numpyArray is not None:
            self.rows = numpyArray.shape[0] if len(numpyArray.shape) > 0 else 1
            self.cols = numpyArray.shape[1] if len(numpyArray.shape) > 1 else 1
            self.dims = numpyArray.shape[2] if len(numpyArray.shape) > 2 else 1
            self.cpp_mat = matTypes.get_lib(self.rows,self.cols,self.dims).Matrix(numpyArray)
            return
        self.rows = rows
        self.cols = cols
        self.dims = dims
        if cpp_mat == None:
            self.cpp_mat = matTypes.get_lib(rows,cols,dims).Matrix()
            return
        else:
            self.cpp_mat = cpp_mat
            return
    




    def Print(self):
        self.cpp_mat.print()
