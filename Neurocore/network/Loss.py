from Neurocore.network.Layers import LayerShape

class Loss:
    def __init__(self):
        pass
    def get_code(self, layerShape: LayerShape):
        pass

class MSE(Loss):
    def __init__(self):
        super().__init__()
    def get_code(self, layerShape: LayerShape):
        return f'MSE<{layerShape.x},{layerShape.y},{layerShape.z}>'