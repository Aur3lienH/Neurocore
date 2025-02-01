


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
        return f'FCL<{self.activation.get_code(self.layerShape,prevLayerShape)},{prevLayerShape.get_code()},{self.layerShape.get_code()}>'
    
    def get_layer_shape(self):
        return self.layerShape

class ConvLayer(Layer):

    def __init__(self, layerShape: LayerShape, activation, kernelShape: LayerShape, optimizer):
        self.layerShape = layerShape
        self.activation = activation
        self.kernelShape = kernelShape
        self.optimizer = optimizer

    def get_code(self, prevLayerShape: LayerShape):
        return f'ConvLayer<{self.activation.get_code(self.layerShape)},{prevLayerShape.get_code()},{self.layerShape.get_code()},{self.optimizer.get_code()}>'

    def get_layer_shape(self):
        return self.layerShape


class Dropout(Layer):
    def __init__(self, layerShape: LayerShape, rate: float):
        self.layerShape = layerShape
        self.rate = rate

    def get_code(self, prevLayerShape: LayerShape):
        return f'Dropout<{self.layerShape.get_code()},{self.rate}>'

    def get_layer_shape(self):
        return self.layerShape


class Reshape(Layer):
    def __init__(self, newShape: LayerShape , prevShape: LayerShape):
        self.newShape = newShape
        self.prevShape = prevShape

    def get_code(self, prevLayerShape: LayerShape):
        return f'Reshape<{self.newShape.get_code()},{self.prevShape.get_code()}>'

    def get_layer_shape(self):
        return self.newShape

class AveragePooling(Layer):

    def __init__(self, layerShape: LayerShape, filterSize, stride):
        self.layerShape = layerShape
        self.filterSize = filterSize
        self.stride = stride

    def get_code(self, prevLayerShape: LayerShape):
        return f'AveragePooling<{self.layerShape.get_code()},{self.filterSize},{self.stride}>'

    def get_layer_shape(self):
        return self.layerShape

class MaxPooling(Layer):

    def __init__(self, layerShape: LayerShape, filterSize, stride):
        self.layerShape = layerShape
        self.filterSize = filterSize
        self.stride = stride

    def get_code(self, prevLayerShape: LayerShape):
        return f'MaxPooling<{self.layerShape.get_code()},{self.filterSize},{self.stride}>'

    def get_layer_shape(self):
        return self.layerShape