from src.network.Network import Network
from src.network.Layers import FCL, InputLayer
from src.network.Activation import ReLU
from src.network.Loss import MSE


net = Network()
net.AddLayer(InputLayer(1))
net.AddLayer(FCL(5,ReLU()))

net.Compile(MSE())
net.Print()

