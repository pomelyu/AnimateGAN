import itertools
import torch
from torch import nn
import numpy as np
from .base_model import BaseModel
from .components.blocks import ConvBlock, DeConvBlock
from .components.layers import get_norm_layer, FlattenLayer, ReshapeLayer, DeConvLayer
from .components.loss import KLLoss
from .util import init_net

# pylint: disable=attribute-defined-outside-init

class VAE(BaseModel):
    def name(self):
        return "VAE"

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--ngf", type=int, default=32, help="# of gen filters in first conv layer")
        parser.add_argument("--latent_size", type=int, default=32)
        if is_train:
            parser.add_argument("--ndf", type=int, default=32, help="# of discrim filters in first conv layer")
            parser.add_argument("--lambda_kl", type=float, default=1.0)
            parser.add_argument("--c_min", type=float, default=0)
            parser.add_argument("--c_max", type=float, default=0)
            parser.add_argument("--c_stop_iter", type=int, default=100000)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.loss_names = ["KL", "Idt", "KL_mean"]
        self.model_names = ["Encoder", "Decoder"]
        self.visual_names = ["real", "fake"]

        self.netEncoder = Encoder(opt.latent_size, ngf=opt.ngf)
        self.netDecoder = Decoder(opt.latent_size, ngf=opt.ngf)

        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netDecoder = init_net(self.netDecoder, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.c = 0
            self.criterionIdt = nn.MSELoss(reduction="sum")
            self.criterionKL = KLLoss()
            params = itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters())
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.real = input_data["image"].to(self.device)
        self.image_paths = input_data["path"]

    def forward(self):
        self.mu, self.logvar = self.netEncoder(self.real)
        z = self.reparameterize(self.mu, self.logvar)
        self.fake = self.netDecoder(z)

    def backward(self):
        # loss_KL is calculated on z, (batch_size, latent_size)
        # loss_Idt is calculated on real/fake, (batch_size, depth, width, height)
        # Hence they should be average by batch_size, not by the number of elements
        num_els = self.real.shape[0]
        loss_KL = self.criterionKL(self.mu, self.logvar)
        self.loss_KL = (loss_KL - self.c).abs() * self.opt.lambda_kl
        self.loss_Idt = self.criterionIdt(self.fake, self.real) / num_els
        self.loss_KL_mean = self.loss_KL / self.mu.shape[1]
        loss = self.loss_Idt + self.loss_KL
        loss.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def update_niter(self, niter):
        opt = self.opt
        self.c = np.interp(niter * opt.batch_size, [0, opt.c_stop_iter], [opt.c_min, opt.c_max])

    @torch.no_grad()
    def get_test_output(self):
        # one sample
        assert self.mu.shape[0] == 1

        z = self.mu
        sigma = self.logvar.exp().sqrt()

        res = []
        for dim in range(z.shape[-1]):
            test_batch = []
            for i in range(-12, 12):
                one_sample = z.clone()
                one_sample[:, dim] = one_sample[:, dim] + sigma[:, dim] * i * 3
                test_batch.append(one_sample)
            test_batch = torch.cat(test_batch, dim=0)
            image_batch = self.netDecoder(test_batch).cpu().numpy()

            # (-1, 1) -> (0, 1)
            image_batch = ((image_batch * 0.5 + 0.5) * 255).astype(np.uint8)
            # (batch, d, h, w) -> (batch, h, w, d)
            image_batch = np.moveaxis(image_batch, 1, -1)
            # (batch, h, w, d) -> (batch * h, w, d)
            image_batch = np.concatenate(image_batch, axis=0)
            res.append(image_batch)

        # (latent, batch * h ,w, d) -> (batch * h, latent * w, d)
        return np.concatenate(res, axis=1)


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
