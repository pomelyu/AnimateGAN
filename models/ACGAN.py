import torch
from torch import nn
from .components.loss import GANLoss
from .components.layers import get_norm_layer
from .components.blocks import ConvBlock
from .util import init_net
from .base_model import BaseModel
from .DCGAN import DCGAN_G

# pylint: disable=attribute-defined-outside-init

class ACGAN(BaseModel):
    def name(self):
        return "ACGAN"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--ngf", type=int, default=128, help="# of gen filters in first conv layer")
        parser.add_argument("--n_hair_color", type=int, default=12)
        parser.add_argument("--n_eyes_color", type=int, default=10)
        if is_train:
            parser.add_argument("--ndf", type=int, default=128, help="# of discrim filters in first conv layer")
            parser.add_argument("--lambda_class", type=float, default=5.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt

        self.loss_names = ["G", "D", "Class_G", "Class_D"]
        self.model_names = ["G"]
        self.visual_names = ["real", "fake"]

        input_nc = opt.latent_size + opt.n_hair_color + opt.n_eyes_color
        self.netG = DCGAN_G(input_nc, 3, ngf=opt.ngf, use_bias=False)
        self.netD = ACGAN_D(3, opt.n_hair_color, opt.n_eyes_color, ndf=opt.ndf, use_bias=False)

        self.netG = init_net(self.netG, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)
        self.netD = init_net(self.netD, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)

        self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
        self.criterionClass = nn.CrossEntropyLoss()
        if self.isTrain:
            self.model_names += ["D"]
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.latent = input_data["latent"].to(self.device)
        self.real = input_data["image"].to(self.device)
        self.hair_color = input_data["hair"].to(self.device)
        self.eyes_color = input_data["eyes"].to(self.device)

    def forward(self):
        latent = torch.cat((self.latent, self.hair_color, self.eyes_color), 1)
        self.fake = self.netG(latent)

    def backward_G(self):
        fake_score, fake_hair, fake_eyes = self.netD(self.fake)

        _, target_hair = self.hair_color.max(-1)
        _, target_eyes = self.eyes_color.max(-1)
        target_hair = target_hair.long()
        target_eyes = target_eyes.long()

        self.loss_G = self.criterionGAN(fake_score, True)
        self.loss_Class_G = self.opt.lambda_class * (
            self.criterionClass(fake_hair, target_hair) +
            self.criterionClass(fake_eyes, target_eyes)
        )
        loss = self.loss_G + self.loss_Class_G
        loss.backward()

    def backward_D(self):
        latent = torch.cat((self.latent, self.hair_color, self.eyes_color), 1)
        fake = self.netG(latent)
        real = self.real

        if self.opt.noise_level > 0:
            fake = fake + self.get_noise_tensor_as(fake)
            real = real + self.get_noise_tensor_as(real)

        fake_score, fake_hair, fake_eyes = self.netD(fake)
        real_score, real_hair, real_eyes = self.netD(real)

        _, target_hair = self.hair_color.max(-1)
        _, target_eyes = self.eyes_color.max(-1)
        target_hair = target_hair.long()
        target_eyes = target_eyes.long()

        self.loss_D = 0.5 * (
            self.criterionGAN(fake_score, False) +
            self.criterionGAN(real_score, True)
        )
        self.loss_Class_D = self.opt.lambda_class * 0.5 * (
            self.criterionClass(fake_hair, target_hair) +
            self.criterionClass(real_hair, target_hair) +
            self.criterionClass(fake_eyes, target_eyes) +
            self.criterionClass(real_eyes, target_eyes)
        )
        loss = self.loss_D + self.loss_Class_D
        loss.backward()


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


# Network Components
class ACGAN_D(nn.Module):
    def __init__(self, input_nc, n_hair_color, n_eyes_color, ndf=64, norm="batch", use_bias=False):
        super(ACGAN_D, self).__init__()

        norm_layer = get_norm_layer(norm)
        self.shared_layers = nn.Sequential(
            # 3, 64, 64 -> 128, 32, 32
            ConvBlock(input_nc, ndf*1, norm_layer=norm_layer, use_bias=use_bias),
            # 128, 32, 32 -> 256, 16, 16
            ConvBlock(ndf*1, ndf*2, norm_layer=norm_layer, use_bias=use_bias),
            # 256, 16, 16 -> 512, 8, 8
            ConvBlock(ndf*2, ndf*4, norm_layer=norm_layer, use_bias=use_bias),
            # 512, 8, 8 -> 1024, 4, 4
            ConvBlock(ndf*4, ndf*8, norm_layer=norm_layer, use_bias=use_bias),
        )
        # output layer
        self.gan_layer = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=use_bias)
        self.hair_layer = nn.Conv2d(ndf*8, n_hair_color, kernel_size=4, stride=1, padding=0, bias=use_bias)
        self.eyes_layer = nn.Conv2d(ndf*8, n_eyes_color, kernel_size=4, stride=1, padding=0, bias=use_bias)

    def forward(self, x): # pylint: disable=arguments-differ
        n = x.shape[0]
        x = self.shared_layers(x)
        return self.gan_layer(x).view(n, -1), self.hair_layer(x).view(n, -1), self.eyes_layer(x).view(n, -1)
