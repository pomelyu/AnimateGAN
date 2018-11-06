from torch import nn
from .util import get_pad_layer

class DeConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, \
        kernel_size=4, stride=2, padding=1, use_bias=False):
        super(DeConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding=padding, bias=use_bias),
            nn.BatchNorm2d(output_nc),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, \
        norm_layer=nn.BatchNorm2d, use_bias=False, pad_type="reflect"):
        super(ConvBlock, self).__init__()

        pad_layer = get_pad_layer(pad_type)
        self.model = nn.Sequential(
            pad_layer(padding),
            nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=use_bias),
            norm_layer(output_nc),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)
