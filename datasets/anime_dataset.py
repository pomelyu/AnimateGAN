from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms as T
from .base_dataset import BaseDataset

# pylint: disable=W0201

class AnimeDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.images = [f for f in Path(opt.dataroot).iterdir() if f.with_suffix(".jpg")]
        self.latent_size = opt.latent_size
        self.transforms = T.Compose([
            T.Resize(opt.load_size),
            T.CenterCrop(opt.crop_size),
            T.ToTensor(),
        ])

    def __getitem__(self, index):
        latent = torch.rand(self.latent_size, 1, 1)

        if not self.opt.isTrain:
            return {"latent": latent}

        image_path = str(self.images[index])
        image = Image.open(image_path)
        image = self.transforms(image)

        return {"latent": latent, "image": image, "path": image_path}

    def __len__(self):
        return len(self.images)

    def name(self):
        return 'AnimeDataset'
