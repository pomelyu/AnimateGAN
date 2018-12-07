import torch
from torch import nn
from .base_model import BaseModel
from .VAE import Encoder, Decoder, KLLoss
from .DCGAN import DCGAN_D
from .building_blocks.loss import GANLoss
from .util import init_net

# pylint: disable=attribute-defined-outside-init

class VAEGAN(BaseModel):
    def name(self):
        return "VAEGAN"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(ngf=32)
        parser.set_defaults(latent_size=256)
        if is_train:
            # weight calculated by observered variance
            parser.add_argument("--lambda_kl", type=float, default=1.0)
            parser.add_argument("--lambda_idt", type=float, default=1.0)
        return parser

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.loss_names = ["KL", "Idt", "G", "D"]
        self.model_names = ["Encoder", "G", "D"]
        self.visual_names = ["real", "fake", "fake_p"]

        self.netEncoder = Encoder(opt.latent_size, ngf=opt.ngf)
        self.netG = Decoder(opt.latent_size, ngf=opt.ngf)
        self.netD = DCGAN_D(3, 1, ndf=opt.ndf, norm="batch", use_bias=False)

        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netG = init_net(self.netG, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netD = init_net(self.netD, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.criterionIdt = nn.MSELoss()
            self.criterionKL = KLLoss()
            self.criterionGAN = GANLoss(use_lsgan=False).to(self.device)
            self.optimizerEn = torch.optim.Adam(self.netEncoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizerEn, self.optimizerG, self.optimizerD]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.real = input_data["image"].to(self.device)

    def forward(self):
        self.mu, self.logvar = self.netEncoder(self.real)
        z = self.reparameterize(self.mu, self.logvar)
        z_p = torch.randn_like(self.mu)
        self.fake = self.netG(z)
        self.fake_p = self.netG(z_p)

    def backward_En(self):
        self.loss_KL = self.criterionKL(self.mu, self.logvar) * self.opt.lambda_kl
        self.loss_Idt = self.criterionIdt(self.fake, self.real) * self.opt.lambda_idt
        loss = self.loss_KL + self.loss_Idt
        loss.backward(retain_graph=True)

    def backward_G(self):
        fake_score = self.netD(self.fake)
        fake_score_p = self.netD(self.fake_p)

        self.loss_G = 0.5 * (self.criterionGAN(fake_score, True) + self.criterionGAN(fake_score_p, True))
        loss_idt = self.criterionIdt(self.fake, self.real) * self.opt.lambda_idt
        loss = self.loss_G + loss_idt
        loss.backward()

    def backward_D(self):
        mu, logvar = self.netEncoder(self.real)
        z = self.reparameterize(mu, logvar)
        z_p = torch.randn_like(mu)
        real_score = self.netD(self.real)
        fake_score = self.netD(self.netG(z))
        fake_score_p = self.netD(self.netG(z_p))

        self.loss_D = (
            self.criterionGAN(real_score, True) + \
            self.criterionGAN(fake_score, False) + \
            self.criterionGAN(fake_score_p, False)
        ) / 3
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad([self.netD], False)
        self.optimizerEn.zero_grad()
        self.backward_En()
        self.optimizerEn.step()

        self.optimizerG.zero_grad()
        self.backward_G()
        self.optimizerG.step()

        self.set_requires_grad([self.netD], True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()
