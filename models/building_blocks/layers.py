import functools
import torch
from torch import nn
import torch.nn.functional as F

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = SkipLayer
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_pad_layer(pad_type="zero"):
    if pad_type == "zero":
        pad_layer = functools.partial(nn.ZeroPad2d)
    elif pad_type == "reflect":
        pad_layer = functools.partial(nn.ReflectionPad2d)
    elif pad_type == "replicate":
        pad_layer = functools.partial(nn.ReplicationPad2d)
    else:
        raise NotImplementedError('padding layer [%s] is not found' % pad_type)
    return pad_layer

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

class FlattenLayer(nn.Module):
    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, -1)

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        num_batch = x.shape[0]
        return x.view(num_batch, *self.shape)

class L2NormalizeLayer(nn.Module):
    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, p=2, dim=1)

class GradientReverseLayer(nn.Module):
    def __init__(self, revsersed_ratio=1):
        super(GradientReverseLayer, self).__init__()
        self.layer = GradientReverse(revsersed_ratio)

    def forward(self, x):
        return self.layer(x)

class GradientReverse(torch.autograd.Function):
    def __init__(self, revsersed_ratio=1):
        super(GradientReverse, self).__init__()
        self.revsersed_ratio = revsersed_ratio

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_out):
        return -self.revsersed_ratio * grad_out.clone()
