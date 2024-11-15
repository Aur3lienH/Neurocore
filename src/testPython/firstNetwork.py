import numpy as np
from libdeep import *
a = np.array([0,1,2,3,4])
b = np.array([0,1,2,3,4])

n = Network()
a = InputLayer(1)
n.AddLayer(a)
n.AddLayer(FullLayer(1,ReLU()))

print("compiling ... !")
n.Compile(Constant,MSE())
print("finished compiling !")
n.Print()
