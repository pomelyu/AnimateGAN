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

class DeConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, use_bias=False):
        super(DeConvLayer, self).__init__()
        self.model = nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, bias=use_bias)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2)
        return self.model(x)
