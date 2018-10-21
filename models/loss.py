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
