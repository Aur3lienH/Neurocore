import numpy as np
from libdeep import *
a = np.array([0,1,2,3,4])
b = np.array([0,1,2,3,4])
c = InputLayer(1)
rel = ReLU()
d = FullLayer(1,rel)
n = Network()
n.AddLayer(c)
n.AddLayer(d)

print("compiling ... !")
n.Compile(Constant,MSE())
print("finished compiling !")
n.Print()
