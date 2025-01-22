from src.network.Network import Network
from src.network.Layers import FCL, InputLayer
from src.network.Activation import ReLU
from src.network.Loss import MSE
from src.network.Matrix import Matrix, NumpyToMatrixArray
import numpy as np


okidoki = np.array([[1,2],[3,4],[5,6]])
print(okidoki.shape)
array = NumpyToMatrixArray(okidoki)


res = Matrix(numpyArray=np.array([0.5,2.5,2.5,3.5,5]))
res.Print()



net = Network()
net.AddLayer(InputLayer(1))
net.AddLayer(FCL(5,ReLU()))



net.Compile(MSE())
net.Print()

forward_input = np.array([0.5])


res = net.FeedForward(forward_input)
print(res)

net.Learn(np.array([[1.0],[2.0]]),np.array([[0.5,2,3,4,5],[7,6,5,4,3]]),1,100000,0.01)
