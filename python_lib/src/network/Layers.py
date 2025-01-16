


class LayerShape:
    def __init__(self,x,y = 1,z = 1,a = 1):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
    def get_code(self):
        return f'LayerShape<{self.x},{self.y},{self.z},{self.a}>'

class Layer:
    def get_code(self,prevLayerShape: LayerShape):
        pass
    def get_layer_shape(self):
        pass


class InputLayer(Layer):
    def __init__(self, neuronsCount: int):
        self.layerShape = LayerShape(neuronsCount)

    def get_code(self, prevLayerShape: LayerShape):
        if prevLayerShape != None:
            pass
        return f'InputLayer<{self.layerShape.get_code()}>'
    
    def get_layer_shape(self):
        return self.layerShape    
        

class FCL(Layer):
    def __init__(self,neuronsCount : int,activation):
        self.activation = activation
        self.layerShape = LayerShape(neuronsCount)

    def get_code(self,prevLayerShape: LayerShape):
        return f'FCL<{self.activation.get_code(self.layerShape)},{prevLayerShape.get_code()},{self.layerShape.get_code()}>'
    
    def get_layer_shape(self):
        return self.layerShape