import torch
import context # pylint: disable=unused-import
from datasets.image2image_dataset import Image2ImageDataset

class Opt():
    dataA = "data/demo/faces"
    dataB = "data/demo/faces"
    load_size = 64
    crop_size = 64
    isTrain = True
    serial_batches = False

def test_image2image_dataset():
    demo_opt = Opt()
    demo_dataset = Image2ImageDataset()
    demo_dataset.initialize(demo_opt)

    res = demo_dataset[0]
    demo_A = res["A"]
    demo_B = res["B"]
    assert torch.is_tensor(demo_A)
    assert demo_A.shape == torch.Tensor(3, demo_opt.crop_size, demo_opt.crop_size).shape
    assert demo_A.min() >= -1 and demo_A.max() <= 1
    assert isinstance(demo_A, torch.FloatTensor)

    assert torch.is_tensor(demo_B)
    assert demo_B.shape == torch.Tensor(3, demo_opt.crop_size, demo_opt.crop_size).shape
    assert demo_B.min() >= -1 and demo_B.max() <= 1
    assert isinstance(demo_B, torch.FloatTensor)

    assert isinstance(res["A_paths"], str)
    assert isinstance(res["B_paths"], str)

if __name__ == "__main__":
    test_image2image_dataset()
