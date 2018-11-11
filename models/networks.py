from torch import nn
from .blocks import DeConvBlock, ConvBlock
from .util import get_norm_layer

class DCGAN_G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=128, use_bias=False):
        super(DCGAN_G, self).__init__()

        self.model = nn.Sequential(
            # 1 x 1 => 4 x 4
            DeConvBlock(input_nc, ngf*8, kernel_size=4, stride=1, padding=0, use_bias=use_bias),
            # 4 x 4 => 8 x 8
            DeConvBlock(ngf*8, ngf*4, use_bias=use_bias),
            # 8 x 8 => 16 x 16
            DeConvBlock(ngf*4, ngf*2, use_bias=use_bias),
            # 16 x 16 => 32 x 32
            DeConvBlock(ngf*2, ngf*1, use_bias=use_bias),
            # 32 x 32 => 64 x 64
            nn.ConvTranspose2d(ngf*1, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.Tanh(),
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1, 1, 1)
        return self.model(x)


class DCGAN_D(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=128, norm="batch", use_lsgan=False, use_bias=False):
        super(DCGAN_D, self).__init__()

        norm_layer = get_norm_layer(norm)
        model = [
            # 64 x 64 => 32 x 32
            ConvBlock(input_nc, ndf*1, norm_layer=norm_layer, use_bias=use_bias),
            # 32 x 32 => 16 x 16
            ConvBlock(ndf*1, ndf*2, norm_layer=norm_layer, use_bias=use_bias),
            # 16 x 16 => 8 x 8
            ConvBlock(ndf*2, ndf*4, norm_layer=norm_layer, use_bias=use_bias),
            # 8 x 8 => 4 x 4
            ConvBlock(ndf*4, ndf*8, norm_layer=norm_layer, use_bias=use_bias),
            # 4 x 4 => 1 x 1
            nn.Conv2d(ndf*8, output_nc, kernel_size=4, stride=1, padding=0, bias=use_bias),
        ]
        if not use_lsgan:
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
