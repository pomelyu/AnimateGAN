import torch
import context # pylint: disable=unused-import
from datasets import AnimeDataset

class Opt():
    dataroot = "data/demo/faces"
    latent_size = 100
    load_size = 96
    crop_size = 96
    isTrain = True

def test_anime_dataset():
    demo_opt = Opt()
    demo_dataset = AnimeDataset()
    demo_dataset.initialize(demo_opt)

    res = demo_dataset[0]
    demo_latent = res["latent"]
    demo_image = res["image"]
    assert torch.is_tensor(demo_latent)
    assert demo_latent.shape == torch.Tensor(demo_opt.latent_size, 1, 1).shape
    assert demo_latent.min() >= 0 and demo_latent.max() <= 1

    assert torch.is_tensor(demo_image)
    assert demo_image.shape == torch.Tensor(3, demo_opt.crop_size, demo_opt.crop_size).shape
    assert demo_image.min() >= 0 and demo_image.max() <= 1

    assert isinstance(res["path"], str)

    print("Test: animate_dataset pass!")

if __name__ == "__main__":
    test_anime_dataset()
