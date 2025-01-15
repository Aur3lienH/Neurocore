from python_lib.src.network.Activation import Activation
from network.Loss import Loss
from network.Layers import Layer

class Network:
    def __init__(self):
        self.layers = []
    
    def AddLayer(self, layer: Layer):
        self.layers.append(layer)

    def Compile(self, loss: Loss):
        self.loss = loss
        for layer in self.layers:
            
