import torch
import context # pylint: disable=unused-import
from models.networks import DCGAN_D, DCGAN_G


def test_DCGAN_G():
    batch_size = 8
    latent_size = 100
    image_size = 64
    generator = DCGAN_G(latent_size, 3)

    x_in = torch.randn(batch_size, latent_size)
    x_out = generator.forward(x_in)

    assert x_out.shape == torch.Tensor(batch_size, 3, image_size, image_size).shape
    assert x_out.min() >= -1 and x_out.max() <= 1

def test_DCGAN_D():
    batch_size = 8
    image_size = 64
    descrminator = DCGAN_D(3, 1)

    x_in = torch.randn(batch_size, 3, image_size, image_size)
    x_out = descrminator.forward(x_in)

    assert x_out.shape == torch.Tensor(batch_size, 1, 1, 1).shape
    assert x_out.min() >= 0 and x_out.max() <= 1


if __name__ == "__main__":
    test_DCGAN_G()
    test_DCGAN_D()
