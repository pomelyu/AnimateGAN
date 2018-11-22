import torch
import numpy as np
import context # pylint: disable=unused-import
from datasets.class2image_dataset import Class2ImageDataset

def assert_onehot_tensor(onehot):
    onehot = onehot.numpy()
    index = np.argsort(onehot)
    assert onehot[index[-1]] == 1
    if len(onehot) > 1:
        assert onehot[index[-2]] == 0
        assert onehot[index[0]] == 0

class Opt():
    dataroot = "data/demo_label"
    latent_size = 100
    load_size = 96
    crop_size = 96
    isTrain = True

def test_class2image_dataset():
    demo_opt = Opt()
    demo_dataset = Class2ImageDataset()
    demo_dataset.initialize(demo_opt)

    res = demo_dataset[0]
    demo_latent = res["latent"]
    demo_image = res["image"]
    demo_hair = res["hair"]
    demo_eyes = res["eyes"]
    assert torch.is_tensor(demo_latent)
    assert demo_latent.shape == torch.Tensor(demo_opt.latent_size).shape
    assert isinstance(demo_latent, torch.FloatTensor)

    assert torch.is_tensor(demo_image)
    assert demo_image.shape == torch.Tensor(3, demo_opt.crop_size, demo_opt.crop_size).shape
    assert demo_image.min() >= -1 and demo_image.max() <= 1
    assert isinstance(demo_image, torch.FloatTensor)

    assert isinstance(res["path"], str)

    assert torch.is_tensor(demo_hair)
    assert isinstance(demo_hair, torch.FloatTensor)
    assert_onehot_tensor(demo_hair)

    assert torch.is_tensor(demo_eyes)
    assert isinstance(demo_eyes, torch.FloatTensor)
    assert_onehot_tensor(demo_eyes)

    print("Test: class2image_dataset pass!")

if __name__ == "__main__":
    test_class2image_dataset()
