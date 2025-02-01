from Neurocore.network.Layers import LayerShape



class Activation:
    def get_code(self,layerShape: LayerShape, prevLayerShape: LayerShape):
        return f'{self.__class__.__name__}<{layerShape.x},{prevLayerShape.x},{layerShape.z},{layerShape.a}>'
        

class LeakyReLU(Activation):
    def __init__(self, leaky_val = 0.01):
        self.leaky_val = leaky_val
        super().__init__(self)
    def get_code(self,layerShape: LayerShape):
        return f'{self.__class__.__name__}<{layerShape.x},{self.leaky_val},{layerShape.y},{layerShape.z},{layerShape.a}>'

class ReLU(Activation):
    pass

class Sigmoid(Activation):
    def __init__(self):
        super.__init__(self)

class SigmoidPrime(Activation):
    def __init__(self):
        super.__init__(self)

class Softmax(Activation):
    def __init__(self):
        super.__init__(self)

class Tanh(Activation):
    def __init__(self):
        super.__init__(self)

