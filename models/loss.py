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
            self.loss = nn.BCELoss()

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

    def discriminator_loss(self, netD, real_data, fake_data):
        real_label = netD(real_data)
        fake_label = netD(fake_data)

        batch_size = real_data.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interp = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()

        interp.requires_grad = True
        interp_label = netD(interp)
        gradient_outputs = torch.ones(interp_label.size()).to(self.device)

        gradient = torch.autograd.grad(outputs=interp_label, inputs=interp, \
                                        grad_outputs=gradient_outputs, only_inputs=True)[0]
        gradient = gradient.view(batch_size, -1)
        gradient_penalty = ((torch.norm(gradient, 2, dim=1) - 1) ** 2).mean()

        return fake_label.mean() - real_label.mean() + self.lambda_gp * gradient_penalty

    def generator_loss(self, fake_label):
        return -fake_label.mean()
