import torch
from torch import nn

# pylint: disable=W0223

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.ones(1) * target_real_label)
        self.register_buffer('fake_label', torch.ones(1) * target_fake_label)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, x, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(x)

    def __call__(self, x, target_is_real):
        target_tensor = self.get_target_tensor(x, target_is_real)
        return self.loss(x, target_tensor)


class WGANGPLoss(nn.Module):
    def __init__(self, lambda_gp, device):
        super(WGANGPLoss, self).__init__()
        self.lambda_gp = lambda_gp
        self.device = device

    def interp_real_fake(self, real, fake):
        batch_size = real.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interp = alpha * real.detach() + (1 - alpha) * fake.detach()
        interp.requires_grad = True
        return interp

    def loss_D(self, real_label, fake_label, interp_label, interp):
        batch_size = real_label.shape[0]
        gradient_outputs = torch.ones(interp_label.shape).to(self.device)
        gradient_outputs.requires_grad = False

        gradient = torch.autograd.grad(outputs=interp_label, inputs=interp, \
                                        grad_outputs=gradient_outputs, only_inputs=True, \
                                        create_graph=True, retain_graph=True)[0]
        gradient = gradient.view(batch_size, -1)
        gradient_penalty = ((torch.norm(gradient, 2, dim=1) - 1) ** 2).mean()

        return fake_label.mean() - real_label.mean() + self.lambda_gp * gradient_penalty

    def loss_G(self, fake_label):
        return -fake_label.mean()


class LatentSimiliarLoss(nn.Module):
    def __init__(self, target=1):
        super(LatentSimiliarLoss, self).__init__()
        self.register_buffer('target', torch.ones(1) * target)
        self.loss = nn.CosineEmbeddingLoss()

    def get_target_tensor(self, x):
        return self.target.expand(1, x.shape[0])

    def __call__(self, x1, x2):
        assert x1.shape == x2.shape
        target_tensor = self.get_target_tensor(x1)
        return self.loss(x1, x2, target_tensor)

class KLLoss(nn.Module):
    def __call__(self, mu, logvar):
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return klds.sum(1).mean(0)
