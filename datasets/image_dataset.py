from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from .base_dataset import BaseDataset

# pylint: disable=W0201
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.suffix == extension for extension in IMG_EXTENSIONS)

class ImageDataset(BaseDataset):
    def name(self):
        return 'ImageDataset'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--dataroot", type=str, default="", help="path to images")
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.data = [f for f in Path(opt.dataroot).iterdir() if is_image_file(f)]
        self.transforms = T.Compose([
            T.Resize(opt.load_size),
            T.CenterCrop(opt.crop_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        image_path = str(self.data[index])
        image = Image.open(image_path)
        image = self.transforms(image)

        return {"image": image, "path": image_path}

    def __len__(self):
        return len(self.data)
