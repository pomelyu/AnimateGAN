from torch import nn
from .blocks import DeConvBlock, ConvBlock
from .layers import DeConvLayer, FlattenLayer, ReshapeLayer, L2NormalizeLayer


class XGAN_DomainEncoder(nn.Module):
    def __init__(self):
        super(XGAN_DomainEncoder, self).__init__()
        self.model = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            ConvBlock(3, 32),
            # (32, 32, 32) -> (64, 16, 16)
            ConvBlock(32, 64),
        )

    def forward(self, x):
        return self.model(x)


class XGAN_SharedEncoder(nn.Module):
    def __init__(self):
        super(XGAN_SharedEncoder, self).__init__()
        self.model = nn.Sequential(
            # (64, 16, 16) -> (128, 8, 8)
            ConvBlock(64, 128),
            # (128, 8, 8) -> (256, 4, 4)
            ConvBlock(128, 256),
            # (256, 4, 4) -> (4096)
            FlattenLayer(),
            # (4096) -> (1024)
            nn.Linear(4096, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            # (1024) -> (1024)
            nn.Linear(1024, 1024, bias=False),
            nn.LeakyReLU(inplace=True),
            # L2 normalize
            L2NormalizeLayer(),
        )

    def forward(self, x):
        return self.model(x)


class XGAN_SharedDecoder(nn.Module):
    def __init__(self):
        super(XGAN_SharedDecoder, self).__init__()
        self.model = nn.Sequential(
            ReshapeLayer(shape=(1024, 1, 1)),
            # (1024, 1, 1) -> (512, 4, 4)
            DeConvBlock(1024, 512, kernel_size=4, stride=1, padding=0),
            # (512, 4, 4) -> (256, 8, 8)
            DeConvBlock(512, 256, method="deConv"),
        )

    def forward(self, x):
        return self.model(x)


class XGAN_DomainDecoder(nn.Module):
    def __init__(self):
        super(XGAN_DomainDecoder, self).__init__()
        self.model = nn.Sequential(
            # (256, 8, 8) -> (128, 16, 16)
            DeConvBlock(256, 128, method="deConv"),
            # (128, 16, 16) -> (64, 32, 32)
            DeConvBlock(128, 64, method="deConv"),
            # (64, 32, 32) -> (3, 64, 64)
            DeConvLayer(64, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


class XGAN_Discriminator(nn.Module):
    def __init__(self):
        super(XGAN_Discriminator, self).__init__()
        self.model = nn.Sequential(
            # (3, 64, 64) -> (16, 32, 32)
            ConvBlock(3, 16),
            # (16, 32, 32) -> (32, 16, 16)
            ConvBlock(16, 32),
            # (32, 16, 16) -> (32, 8, 8)
            ConvBlock(32, 32),
            # (32, 8, 8) -> (32, 4, 4)
            ConvBlock(32, 32),
            # (32, 4, 4) -> (512)
            FlattenLayer(),
            # (512) -> (1)
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.model(x)
