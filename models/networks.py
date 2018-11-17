from torch import nn
from .blocks import DeConvBlock, ConvBlock
from .util import get_norm_layer
from .layers import DeConvLayer

class DCGAN_G(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=128, n_layers=5, use_bias=False):
        super(DCGAN_G, self).__init__()
        assert n_layers >= 2

        # 1st layer
        nc_out = ngf*8
        model = [DeConvBlock(input_nc, nc_out, kernel_size=4, stride=1, padding=0, use_bias=use_bias)]

        # middle layer
        for i in reversed(range(0, n_layers-2)):
            nc_in = nc_out
            nc_out = nc_out if i >= 3 else nc_in // 2
            model.append(DeConvBlock(nc_in, nc_out, method="deConv"))

        # output layer
        model += [
            DeConvLayer(nc_out, output_nc),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(x.size()[0], -1, 1, 1)
        return self.model(x)


class DCGAN_D(nn.Module):
    def __init__(self, input_nc, output_nc, ndf=128, n_layers=5, norm="batch", use_bias=False):
        super(DCGAN_D, self).__init__()
        assert n_layers >= 2

        norm_layer = get_norm_layer(norm)

        # 1st layer
        nc_out = ndf*1
        model = [ConvBlock(input_nc, nc_out, norm_layer=norm_layer, use_bias=use_bias)]

        # middle layer
        for _ in range(0, n_layers-2):
            nc_in = nc_out
            nc_out = min(ndf*8, nc_in*2)
            model.append(ConvBlock(nc_in, nc_out, norm_layer=norm_layer, use_bias=use_bias))

        # output layer
        model.append(nn.Conv2d(ndf*8, output_nc, kernel_size=4, stride=1, padding=0, bias=use_bias))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
