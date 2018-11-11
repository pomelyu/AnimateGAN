from torch import nn
import torch.nn.functional as F

class SkipLayer(nn.Module):
    def __init__(self, *args): # pylint: disable=unused-argument
        super(SkipLayer, self).__init__()

    def forward(self, x):
        return x

class InterpolateLayer(nn.Module):
    def __init__(self, scale_factor):
        super(InterpolateLayer, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
