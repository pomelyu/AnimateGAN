import torch
from .base_model import BaseModel
from .networks import BasicD, BasicG
from .loss import GANLoss
from .util import init_net

# pylint: disable=W0201

class AnimeGANModel(BaseModel):
    def name(self):
        return "AnimeGANModel"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        if is_train:
            parser.add_argument("--every_g", default=5)
            parser.add_argument("--every_d", default=1)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.iters = 1

        self.loss_names = ["D", "G"]
        self.model_names = ["D", "G"]
        self.visual_names = ["true", "fake"]

        self.netG = BasicG(opt.latent_size, 3, ngf=opt.ngf, use_bias=False)
        self.netD = BasicD(3, 1, ndf=opt.ndf, use_bias=False)

        self.netG = init_net(self.netG, init_type="xavier", gpu_ids=opt.gpu_ids)
        self.netD = init_net(self.netD, init_type="xavier", gpu_ids=opt.gpu_ids)

        self.criterionGAN = GANLoss(use_lsgan=True)
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input_data):
        self.latent = input_data["latent"]
        self.true = input_data["image"]

    def forward(self):
        self.fake = self.netG(self.latent)

    def backward_G(self):
        self.loss_G = self.criterionGAN(self.netD(self.fake), True)
        self.loss_G.backward()

    def backward_D(self):
        fake = self.netG(self.latent)
        self.loss_D = self.criterionGAN(self.netD(fake), False) + \
            self.criterionGAN(self.netD(self.true), True)
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
