import torch
from .components.loss import WGANGPLoss
from .util import init_net
from .ACGAN import ACGAN, ACGAN_D

# pylint: disable=attribute-defined-outside-init

class ACGANGP(ACGAN):
    def name(self):
        return "ACGANGP"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        ACGAN.modify_commandline_options(parser, is_train)
        if is_train:
            parser.set_defaults(beta1=0)
            parser.add_argument("--every_g", type=int, default=5)
            parser.add_argument("--lambda_gp", type=float, default=10.0)
        return parser

    def initialize(self, opt):
        ACGAN.initialize(self, opt)

        self.niters = 0
        self.netD = ACGAN_D(3, opt.n_hair_color, opt.n_eyes_color, norm="none", ndf=opt.ndf, use_bias=False)
        self.netD = init_net(self.netD, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)

        self.criterionGAN = WGANGPLoss(opt.lambda_gp, self.device).to(self.device)
        if self.isTrain:
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def backward_G(self):
        fake_score, fake_hair, fake_eyes = self.netD(self.fake)

        _, target_hair = self.hair_color.max(-1)
        _, target_eyes = self.eyes_color.max(-1)
        target_hair = target_hair.long()
        target_eyes = target_eyes.long()

        self.loss_G = self.criterionGAN.loss_G(fake_score)
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
        interp = self.criterionGAN.interp_real_fake(real, fake)

        fake_score, fake_hair, fake_eyes = self.netD(fake)
        real_score, real_hair, real_eyes = self.netD(real)
        interp_score, _, _ = self.netD(interp)

        _, target_hair = self.hair_color.max(-1)
        _, target_eyes = self.eyes_color.max(-1)
        target_hair = target_hair.long()
        target_eyes = target_eyes.long()

        self.loss_D = self.criterionGAN.loss_D(real_score, fake_score, interp_score, interp)
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

        if self.niters % self.opt.every_g == 0:
            self.set_requires_grad([self.netD], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            self.niters = 0

        self.niters += 1
