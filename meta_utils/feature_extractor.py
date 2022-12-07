# General feature extractor based on:
# https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
import torch

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers, runner=None):
        super().__init__()
        self.model = model
        self.layers = layers
        self.runner = runner
        if runner is None:
            self.runner = self.model
        self._features = []
        for layer_id, layer in enumerate(layers):
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id):
        def fn(_, __, output):
            self._features.append(output)
        return fn

    def forward(self, x, **kwargs):
        self._features = []
        with torch.no_grad():
            _ = self.runner(x, **kwargs)
        return self._features