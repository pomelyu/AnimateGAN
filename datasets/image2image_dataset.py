import random
from PIL import Image
from torchvision import transforms as T
from .base_dataset import BaseDataset
from .image_folder import make_dataset

class Image2ImageDataset(BaseDataset):
    def name(self):
        return "Image2ImageDataset"

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--dataA", type=str, required=True)
        parser.add_argument("--dataB", type=str, required=True)
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.A_paths = sorted(make_dataset(opt.dataA))
        self.B_paths = sorted(make_dataset(opt.dataB))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transforms = T.Compose([
            T.Resize(opt.load_size),
            T.CenterCrop(opt.crop_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            B_path = self.B_paths[index % self.B_size]
        else:
            B_path = self.B_paths[random.randint(0, self.B_size-1)]

        A_image = Image.open(A_path)
        B_image = Image.open(B_path)

        A = self.transforms(A_image)
        B = self.transforms(B_image)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
