import itertools
import torch
from torch import nn
from .building_blocks.blocks import DeConvBlock, ConvBlock
from .building_blocks.layers import DeConvLayer, FlattenLayer, ReshapeLayer, L2NormalizeLayer, GradientReverseLayer
from .building_blocks.loss import GANLoss, LatentSimiliarLoss
from .util import init_net
from .base_model import BaseModel

class XGAN(BaseModel):
    def name(self):
        return "XGAN"

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataroot="no effect")
        parser.add_argument("--direction", type=str, default="AtoB", help="AtoB, BtoA")
        parser.add_argument("--lambda_dann", type=float, default=1.0)
        parser.add_argument("--lambda_sem", type=float, default=1.0)
        parser.add_argument("--lambda_rec", type=float, default=1.0)
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.opt = opt
        self.loss_names = ["G_A", "idt_A", "sem_A", "D_B", "idt_B", "sem_B", "dann"]
        self.model_names = ["En_A", "En_B", "De_A", "De_B", "En_Shared", "De_Shared"]
        if opt.isTrain:
            self.model_names += ["D_B", "LC"]
        self.visual_names = ["real_A", "real_B", "fake_A", "fake_B", "rec_A", "rec_B"]

        self.netEn_A = XGAN_DomainEncoder()
        self.netEn_B = XGAN_DomainEncoder()
        self.netDe_A = XGAN_DomainDecoder()
        self.netDe_B = XGAN_DomainDecoder()
        self.netEn_Shared = XGAN_SharedEncoder()
        self.netDe_Shared = XGAN_SharedDecoder()

        self.criterionGAN = GANLoss().to(self.device)
        self.criterionL1 = nn.L1Loss().to(self.device)
        self.criterionLatent = LatentSimiliarLoss().to(self.device)

        if opt.isTrain:
            # self.netD_A = XGAN_Discriminator()
            self.netD_B = XGAN_Discriminator()
            self.netLC = XGAN_LatentClassifer()
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                self.netEn_A.parameters(),
                self.netEn_B.parameters(),
                self.netEn_Shared.parameters(),
                self.netDe_Shared.parameters(),
                self.netDe_A.parameters(),
                self.netDe_B.parameters(),
            ), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(
                # self.netD_A.parameters(),
                self.netD_B.parameters(),
            ), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_Domain = torch.optim.Adam(itertools.chain(
                self.netEn_A.parameters(),
                self.netEn_B.parameters(),
                self.netEn_Shared.parameters(),
                self.netLC.parameters(),
            ), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [
                self.optimizer_G,
                self.optimizer_D,
                self.optimizer_Domain,
            ]

        for name in self.model_names:
            net = getattr(self, "net" + name)
            net = init_net(net, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            setattr(self, "net" + name, net)

    def set_input(self, input_data):
        self.real_A = input_data["A"].to(self.device)
        self.real_B = input_data["B"].to(self.device)

    def forward(self):
        self.latent_A = self.netEn_Shared(self.netEn_A(self.real_A))
        self.latent_B = self.netEn_Shared(self.netEn_B(self.real_B))
        self.rec_A = self.netDe_A(self.netDe_Shared(self.latent_A))
        self.rec_B = self.netDe_B(self.netDe_Shared(self.latent_B))
        self.fake_B = self.netDe_B(self.netDe_Shared(self.latent_A))
        self.fake_A = self.netDe_A(self.netDe_Shared(self.latent_B))
        self.sem_A = self.netEn_Shared(self.netEn_A(self.fake_A))
        self.sem_B = self.netEn_Shared(self.netEn_B(self.fake_B))

    def backward_G(self):
        self.loss_G_A = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_idt_A = self.criterionL1(self.real_A, self.rec_A) * self.opt.lambda_rec / self.real_A.numel()
        self.loss_idt_B = self.criterionL1(self.real_B, self.rec_B) * self.opt.lambda_rec / self.real_B.numel()
        self.loss_sem_A = self.criterionLatent(self.latent_A, self.sem_B) * self.opt.lambda_sem
        self.loss_sem_B = self.criterionLatent(self.latent_B, self.sem_A) * self.opt.lambda_sem

        total_loss = self.loss_G_A + self.loss_idt_A + self.loss_idt_B + self.loss_sem_A + self.loss_sem_B
        total_loss.backward()

    def backward_Domain(self):
        latent_A = self.netEn_Shared(self.netEn_A(self.real_A))
        latent_B = self.netEn_Shared(self.netEn_B(self.real_B))

        self.loss_dann = self.criterionGAN(self.netLC(latent_A), True) + \
                         self.criterionGAN(self.netLC(latent_B), False)

        self.loss_dann = self.loss_dann * self.opt.lambda_dann
        self.loss_dann.backward()

    def backward_D(self):
        latent_A = self.netEn_Shared(self.netEn_A(self.real_A))
        fake_B = self.netDe_B(self.netDe_Shared(latent_A))
        self.loss_D_B = (self.criterionGAN(self.netD_B(fake_B), False) + \
                        self.criterionGAN(self.netD_B(self.real_B), True)) * 0.5
        self.loss_D_B.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.optimizer_Domain.zero_grad()
        self.backward_Domain()
        self.optimizer_Domain.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()


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


class XGAN_LatentClassifer(nn.Module):
    def __init__(self):
        super(XGAN_LatentClassifer, self).__init__()
        self.model = nn.Sequential(
            GradientReverseLayer(1),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
