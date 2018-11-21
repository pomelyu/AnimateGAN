import torch
import context # pylint: disable=unused-import
from models import XGAN

def assertEqual(t1, t2):
    assert torch.all(torch.lt(torch.abs(torch.add(t1, -t2)), 1e-6))

def test_encoder():
    batch_size = 8
    latent_size = 1024
    x = torch.rand((batch_size, 3, 64, 64))*2 - 1

    domain_encoder = XGAN.XGAN_DomainEncoder()
    shared_encoder = XGAN.XGAN_SharedEncoder()

    x = domain_encoder(x)
    x = shared_encoder(x)

    assert x.shape == torch.Tensor(batch_size, latent_size).shape
    assertEqual(torch.norm(x, p=2, dim=1), torch.ones(batch_size))

def test_decoder():
    batch_size = 8
    latent_size = 1024
    x = torch.rand((batch_size, latent_size))

    shared_decoder = XGAN.XGAN_SharedDecoder()
    domain_decoder = XGAN.XGAN_DomainDecoder()

    x = shared_decoder(x)
    x = domain_decoder(x)

    assert x.shape == torch.Tensor(batch_size, 3, 64, 64).shape
    assert x.min() >= -1 and x.min() < 0
    assert x.max() <= 1 and x.max() > 0

def test_discriminator():
    batch_size = 8
    x = torch.rand((batch_size, 3, 64, 64))*2 - 1

    discriminator = XGAN.XGAN_Discriminator()

    x = discriminator(x)

    assert x.shape == torch.Tensor(batch_size, 1).shape

if __name__ == "__main__":
    test_encoder()
    test_decoder()
    test_discriminator()
