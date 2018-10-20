import torch
import context # pylint: disable=unused-import
from models.blocks import DeConvBlock, ConvBlock


def test_DeConvBlock():
    batch_size = 8
    dim = 64
    input_nc = 16
    output_nc = 32
    deConv = DeConvBlock(input_nc, output_nc, scale_factor=2, use_bias=False)

    x_in = torch.randn(batch_size, input_nc, dim, dim)
    x_out = deConv.forward(x_in)

    assert x_out.shape == torch.Tensor(batch_size, output_nc, dim * 2, dim * 2).shape

def test_ConvBlock():
    batch_size = 8
    dim = 64
    input_nc = 16
    output_nc = 32
    deConv = ConvBlock(input_nc, output_nc, use_bias=False)

    x_in = torch.randn(batch_size, input_nc, dim, dim)
    x_out = deConv.forward(x_in)

    assert x_out.shape == torch.Tensor(batch_size, output_nc, dim // 2, dim // 2).shape


if __name__ == "__main__":
    test_DeConvBlock()
    test_ConvBlock()
