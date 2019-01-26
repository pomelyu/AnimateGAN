import torch
from torch import nn
from .base_model import BaseModel
from .VAE import Encoder, Decoder, KLLoss
from .DCGAN import DCGAN_D
from .components.loss import WGANGPLoss
from .util import init_net

# pylint: disable=attribute-defined-outside-init

class VAEGAN(BaseModel):
    def name(self):
        return "VAEGAN"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(ngf=32)
        parser.set_defaults(latent_size=256)
        parser.set_defaults(beta1=0)
        if is_train:
            # weight calculated by observered variance
            parser.add_argument("--lambda_kl", type=float, default=1.0)
            parser.add_argument("--lambda_idt", type=float, default=1.0)
            parser.add_argument("--lambda_gp", type=float, default=10.0)
            parser.add_argument("--every_g", type=int, default=5)
        return parser

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.niters = 0
        self.loss_names = ["KL", "Idt", "G", "D"]
        self.model_names = ["Encoder", "G", "D"]
        self.visual_names = ["real", "fake", "fake_p"]

        self.netEncoder = Encoder(opt.latent_size, ngf=opt.ngf)
        self.netG = Decoder(opt.latent_size, ngf=opt.ngf)
        self.netD = DCGAN_D(3, 1, ndf=opt.ndf, norm="none", use_bias=False)

        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netG = init_net(self.netG, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netD = init_net(self.netD, opt.init_type, opt.init_gain, opt.gpu_ids)

        if self.isTrain:
            self.criterionIdt = nn.MSELoss()
            self.criterionKL = KLLoss()
            self.criterionGAN = WGANGPLoss(opt.lambda_gp, self.device).to(self.device)
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

        self.loss_G = 0.5 * (
            self.criterionGAN.loss_G(fake_score) + \
            self.criterionGAN.loss_G(fake_score_p)
        )
        loss_idt = self.criterionIdt(self.fake, self.real) * self.opt.lambda_idt
        loss = self.loss_G + loss_idt
        loss.backward()

    def backward_D(self):
        mu, logvar = self.netEncoder(self.real)
        z = self.reparameterize(mu, logvar)
        z_p = torch.randn_like(mu)
        fake = self.netG(z)
        fake_p = self.netG(z_p)
        real_score = self.netD(self.real)
        interp = self.criterionGAN.interp_real_fake(self.real, fake)
        interp_p = self.criterionGAN.interp_real_fake(self.real, fake_p)

        self.loss_D = 0.5 * (
            self.criterionGAN.loss_D(real_score, self.netD(fake), self.netD(interp), interp) + \
            self.criterionGAN.loss_D(real_score, self.netD(fake_p), self.netD(interp_p), interp_p)
        )
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        if self.niters % self.opt.every_g == 0:
            self.set_requires_grad([self.netD], False)
            self.optimizerEn.zero_grad()
            self.backward_En()
            self.optimizerEn.step()

            self.optimizerG.zero_grad()
            self.backward_G()
            self.optimizerG.step()
            self.niters = 0

        self.set_requires_grad([self.netD], True)
        self.optimizerD.zero_grad()
        self.backward_D()
        self.optimizerD.step()

        self.niters += 1
