from torch import nn
from .blocks import ConvBlock, DeConvBlock, Interpolate

class BasicG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_bias=False):
        super(BasicG, self).__init__()

        self.model = nn.Sequential(
            # 1 x 1 => 4 x 4
            DeConvBlock(input_nc, ngf*8, scale_factor=4, use_bias=use_bias),
            # 4 x 4 => 8 x 8
            DeConvBlock(ngf*8, ngf*4, use_bias=use_bias),
            # 8 x 8 => 16 x 16
            DeConvBlock(ngf*4, ngf*2, use_bias=use_bias),
            # 16 x 16 => 32 x 32
            DeConvBlock(ngf*2, ngf*1, use_bias=use_bias),
            # 32 x 32 => 96 x 96
            Interpolate(scale_factor=3),
            nn.ZeroPad2d(2),
            nn.Conv2d(ngf*1, output_nc, kernel_size=5, padding=2, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1, 1, 1)
        return self.model(x)


class BasicD(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=64, use_bias=False):
        super(BasicD, self).__init__()

        self.model = nn.Sequential(
            # 96 x 96 => 32 x 32
            ConvBlock(input_nc, ndf*1, kernel_size=5, stride=3, padding=1, use_bias=use_bias),
            # 32 x 32 => 16 x 16
            ConvBlock(ndf*1, ndf*2, use_bias=use_bias),
            # 16 x 16 => 8 x 8
            ConvBlock(ndf*2, ndf*4, use_bias=use_bias),
            # 8 x 8 => 4 x 4
            ConvBlock(ndf*4, ndf*8, use_bias=use_bias),
            # 4 x 4 => 1 x 1
            nn.Conv2d(ndf*8, output_nc, kernel_size=4, stride=1, padding=0, bias=use_bias),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
