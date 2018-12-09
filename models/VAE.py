import itertools
import torch
from torch import nn
from .base_model import BaseModel
from .building_blocks.blocks import ConvBlock, DeConvBlock
from .building_blocks.layers import get_norm_layer, FlattenLayer, ReshapeLayer, DeConvLayer
from .building_blocks.loss import KLLoss
from .util import init_net

# pylint: disable=attribute-defined-outside-init

class VAE(BaseModel):
    def name(self):
        return "VAE"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(ngf=32)
        parser.set_defaults(latent_size=256)
        if is_train:
            # weight calculated by observered variance
            parser.add_argument("--lambda_kl", type=float, default=1.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.loss_names = ["KL", "Idt", "All"]
        self.model_names = ["Encoder", "Decoder"]
        self.visual_names = ["real", "fake"]

        self.netEncoder = Encoder(opt.latent_size, ngf=opt.ngf)
        self.netDecoder = Decoder(opt.latent_size, ngf=opt.ngf)

        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netDecoder = init_net(self.netDecoder, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.criterionIdt = nn.MSELoss()
            self.criterionKL = KLLoss()
            params = itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.real = input_data["image"].to(self.device)

    def forward(self):
        self.mu, self.logvar = self.netEncoder(self.real)
        z = self.reparameterize()
        self.fake = self.netDecoder(z)

    def reparameterize(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + self.mu

    def backward(self):
        self.loss_KL = self.criterionKL(self.mu, self.logvar) * self.opt.lambda_kl
        self.loss_Idt = self.criterionIdt(self.fake, self.real)
        self.loss_All = self.loss_Idt + self.loss_KL
        self.loss_All.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()


class Encoder(nn.Module):
    def __init__(self, latent_size, ngf=128, norm="batch", use_bias=False):
        super(Encoder, self).__init__()
        norm_layer = get_norm_layer(norm)
        self.model = nn.Sequential(
            # 3, 64, 64 -> 32, 32, 32
            ConvBlock(3, ngf*1, norm_layer=norm_layer, use_bias=use_bias),
            # 32, 32, 32 -> 64, 16, 16
            ConvBlock(ngf*1, ngf*2, norm_layer=norm_layer, use_bias=use_bias),
            # 64, 16, 16 -> 128, 8, 8
            ConvBlock(ngf*2, ngf*4, norm_layer=norm_layer, use_bias=use_bias),
            # 128, 8, 8 -> 256, 4, 4
            ConvBlock(ngf*4, ngf*8, norm_layer=norm_layer, use_bias=use_bias),
            # 256, 4, 4 -> 256, 2, 2
            ConvBlock(ngf*8, ngf*8, norm_layer=norm_layer, use_bias=use_bias),
            # 256, 2, 2 -> 1024
            FlattenLayer(),
        )
        self.mu_output = nn.Linear(1024, latent_size)
        self.logvar_output = nn.Linear(1024, latent_size)

    def forward(self, x):
        x = self.model(x)
        return self.mu_output(x), self.logvar_output(x)


class Decoder(nn.Module):
    def __init__(self, latent_size, ngf=128, norm="batch", use_bias=False):
        super(Decoder, self).__init__()
        norm_layer = get_norm_layer(norm)
        self.model = nn.Sequential(
            # latent -> 1024
            nn.Linear(latent_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # 1024 -> 256, 2, 2
            ReshapeLayer((256, 2, 2)),
            # 256, 2, 2 -> 256, 4, 4
            DeConvBlock(ngf*8, ngf*8, method="deConv", norm_layer=norm_layer, use_bias=use_bias),
            # 256, 4, 4 -> 128, 8, 8
            DeConvBlock(ngf*8, ngf*4, method="deConv", norm_layer=norm_layer, use_bias=use_bias),
            # 128, 8, 8 -> 64, 16, 16
            DeConvBlock(ngf*4, ngf*2, method="deConv", norm_layer=norm_layer, use_bias=use_bias),
            # 64, 16, 16 -> 32, 32, 32
            DeConvBlock(ngf*2, ngf*1, method="deConv", norm_layer=norm_layer, use_bias=use_bias),
            # 32, 32, 32 -> 3, 64, 64
            DeConvLayer(ngf*1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)
