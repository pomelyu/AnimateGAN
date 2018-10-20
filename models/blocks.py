from torch import nn
import torch.nn.functional as F

class DeConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, scale_factor=2, kernel_size=3, padding=1, stride=1, use_bias=False):
        super(DeConvBlock, self).__init__()

        self.scale_factor = scale_factor
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, bias=use_bias),
            nn.BatchNorm2d(output_nc),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor)
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, padding=1, stride=2, use_bias=False):
        super(ConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, bias=use_bias),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super(Interpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)
