from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T
from .base_dataset import BaseDataset

# pylint: disable=W0201

class Class2ImageDataset(BaseDataset):
    def name(self):
        return 'Class2ImageDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--dataroot", required=True, help="path to folder")
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dataroot = Path(opt.dataroot)
        self.data = [f.stem for f in (self.dataroot / "labels").iterdir() if f.suffix == ".txt"]
        self.latent_size = opt.latent_size
        self.transforms = T.Compose([
            T.Resize(opt.load_size),
            T.CenterCrop(opt.crop_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        file_name = self.data[index]
        label_path = self.dataroot / "labels" / "{}.txt".format(file_name)
        labels = self._load_label(label_path)

        latent = np.random.normal(0, 1, size=self.latent_size).astype(np.float32)
        latent = torch.from_numpy(latent)

        image_path = self.dataroot / "images" / "{}.jpg".format(file_name)
        image = Image.open(str(image_path))
        image = self.transforms(image)

        return {
            "latent": latent,
            "image": image,
            "path": str(image_path),
            "hair": torch.FloatTensor(labels[0]),
            "eyes": torch.FloatTensor(labels[1]),
        }

    def __len__(self):
        return len(self.data)

    def _load_label(self, path):
        labels = []
        with path.open('r') as f:
            for line in f:
                labels.append([int(n) for n in line.split(",")])

        return labels
