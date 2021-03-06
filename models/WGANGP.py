import torch
from .base_model import BaseModel
from .components.loss import WGANGPLoss
from .util import init_net
from .DCGAN import DCGAN_G, DCGAN_D

# pylint: disable=attribute-defined-outside-init

class WGANGP(BaseModel):
    def name(self):
        return "WGANGP"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--ngf", type=int, default=128, help="# of gen filters in first conv layer")
        if is_train:
            parser.add_argument("--ndf", type=int, default=128, help="# of discrim filters in first conv layer")
            parser.add_argument("--lambda_gp", type=float, default=10.0)
            parser.add_argument("--every_g", type=int, default=5)
            parser.add_argument("--every_d", type=int, default=1)
            parser.set_defaults(beta1=0)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.iters = 1

        self.loss_names = ["G", "D"]
        self.model_names = ["G", "D"]
        self.visual_names = ["real", "fake"]

        self.netG = DCGAN_G(opt.latent_size, 3, ngf=opt.ngf, use_bias=False)
        self.netD = DCGAN_D(3, 1, norm="none", ndf=opt.ndf, use_bias=False)

        self.netG = init_net(self.netG, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)
        self.netD = init_net(self.netD, init_type="normal", init_gain=0.02, gpu_ids=opt.gpu_ids)

        self.criterionGANGP = WGANGPLoss(opt.lambda_gp, self.device).to(self.device)
        if self.opt.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input_data): # pylint: disable=arguments-differ
        self.latent = input_data["latent"].to(self.device)
        self.real = input_data["image"].to(self.device)

    def forward(self):
        self.fake = self.netG(self.latent)

    def backward_G(self):
        self.loss_G = self.criterionGANGP.loss_G(self.netD(self.fake))
        self.loss_G.backward()

    def backward_D(self):
        fake = self.netG(self.latent)
        real = self.real
        interp = self.criterionGANGP.interp_real_fake(real, fake)
        self.loss_D = self.criterionGANGP.loss_D(self.netD(real), self.netD(fake), self.netD(interp), interp)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        if self.iters % self.opt.every_d == 0:
            self.set_requires_grad([self.netD], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        if self.iters % self.opt.every_g == 0:
            self.set_requires_grad([self.netD], False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

        self.iters += 1

    def evaluate(self):
        with torch.no_grad():
            self.forward()
            return self.criterionGANGP.generator_loss(self.netD(self.fake))
