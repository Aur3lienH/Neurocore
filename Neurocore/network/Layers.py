from typing import LiteralString


class LayerShape:
    x: int
    y: int
    z: int
    a: int

    def __init__(self,x:int, y:int = 1,z:int = 1, a:int = 1):
        self.x = x
        self.y = y
        self.z = z
        self.a = a
    def get_code(self):
        return f'LayerShape<{self.x},{self.y},{self.z},{self.a}>'

    def to_string(self) -> str:
        str_res = '(' + str(self.x)
        if self.y > 1:
            str_res += ', ' + str(self.y)
        if self.z > 1:
            str_res += ', ' + str(self.z)
        if self.a > 1:
            str_res += ', ' + str(self.a)
        str_res += ')'
        return str_res

class Layer:
    def get_code(self,prevLayerShape: LayerShape):
        pass
    def get_layer_shape(self):
        pass


class InputLayer(Layer):
    def __init__(self, neuronsCount: int = None, layerShape: LayerShape = None):
        if neuronsCount is None and layerShape is None:
            raise ValueError("Should either give the neuronsCount or the layerShape in the InputLayer !")
        if neuronsCount is not None and layerShape is not None:
            raise ValueError("Should either give the neuronsCount or the layerShape in the InputLayer !")
        if neuronsCount is None:
            self.layerShape = layerShape
        if layerShape is None:
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

    def __init__(self, layerShape: LayerShape, activation, kernelShape: LayerShape):
        self.layerShape = layerShape
        self.activation = activation
        self.kernelShape = kernelShape

    def get_code(self, prevLayerShape: LayerShape):
        exep_dim_x, exep_dim_y, exep_dim_z = prevLayerShape.x - self.kernelShape.x + 1, prevLayerShape.y - self.kernelShape.y + 1, self.layerShape.z
        if exep_dim_x != self.layerShape.x or exep_dim_y != self.layerShape.y or exep_dim_z == 0:
            expect = LayerShape(exep_dim_x, exep_dim_y, exep_dim_z)
            raise ValueError('ConvLayer: Layer shape is ' + self.layerShape.to_string() + 'But should be ' + expect.to_string() + ' !')
        return f'ConvLayer<{self.activation.get_code(self.layerShape,prevLayerShape)},{prevLayerShape.get_code()},{self.layerShape.get_code()},{self.kernelShape.get_code()}>'

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
        sum_new_shape = self.newShape.x * self.newShape.y * self.newShape.z
        sum_prev_shape = self.prevShape.x * self.prevShape.y * self.prevShape.z
        if sum_new_shape != sum_prev_shape:
            raise ValueError('Reshape: Incorrect number of input and output neurons : the number of input neurons is ' + str(sum_prev_shape) + ' and output neurons : ' + str(sum_new_shape) + ' !')
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

    def __init__(self, layerShape: LayerShape, filterSize: int, stride: int):
        self.layerShape = layerShape
        self.filterSize = filterSize
        self.stride = stride

    def get_code(self, prevLayerShape: LayerShape):
        if prevLayerShape.z != self.layerShape.z:
            raise ValueError('MaxPooling: Number of input and output channels must be the same !')
        return f'MaxPoolLayer<{self.layerShape.get_code()},{prevLayerShape.get_code()},{self.filterSize},{self.stride}>'

    def get_layer_shape(self):
        return self.layerShape