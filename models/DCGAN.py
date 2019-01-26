import random
import torch
from torch import nn
from .base_model import BaseModel
from .components.loss import GANLoss
from .components.layers import get_norm_layer, DeConvLayer
from .components.blocks import DeConvBlock, ConvBlock
from .util import init_net

# pylint: disable=attribute-defined-outside-init

class DCGAN(BaseModel):
    def name(self):
        return "DCGAN"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(ngf=128)
        parser.set_defaults(ndf=128)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt

        self.loss_names = ["G", "D"]
        self.model_names = ["G", "D"]
        self.visual_names = ["real", "fake"]

        self.netG = DCGAN_G(opt.latent_size, 3, ngf=opt.ngf, use_bias=False)
        self.netD = DCGAN_D(3, 1, ndf=opt.ndf, use_bias=False)

        self.netG = init_net(self.netG, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)
        self.netD = init_net(self.netD, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)

        self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.latent = input_data["latent"].to(self.device)
        self.real = input_data["image"].to(self.device)

    def forward(self):
        self.fake = self.netG(self.latent)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.netD(self.fake), True)
        self.loss_G.backward()

    def backward_D(self):
        fake = self.netG(self.latent)
        real = self.real
        if self.opt.noise_level > 0:
            fake = fake + self.get_noise_tensor_as(fake)
            real = real + self.get_noise_tensor_as(real)

        if random.random() <= self.opt.flip_prob:
            self.loss_D = self.criterionGAN(self.netD(fake), True) + self.criterionGAN(self.netD(real), False)
        else:
            self.loss_D = self.criterionGAN(self.netD(fake), False) + self.criterionGAN(self.netD(real), True)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad([self.netD], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def evaluate(self):
        with torch.no_grad():
            self.forward()
            return self.criterionGAN(self.netD(self.fake), True)



# Network Components
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

    def forward(self, x): # pylint: disable=arguments-differ
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

    def forward(self, x): # pylint: disable=arguments-differ
        return self.model(x)
