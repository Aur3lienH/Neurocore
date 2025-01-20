from src.network.Network import Network
from src.network.Layers import FCL, InputLayer
from src.network.Activation import ReLU
from src.network.Loss import MSE
from src.network.Matrix import Matrix


mat = Matrix(2,2,1)
mat2 = Matrix(2,2,12)
mat3 = Matrix(2,100,1)
mat5 = Matrix(199,2,1)
mat.Print()
mat2.Print()

net = Network()
net.AddLayer(InputLayer(1))
net.AddLayer(FCL(5,ReLU()))

net.Compile(MSE())
net.Print()

