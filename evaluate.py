import math
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image

from options.eval_options import EvalOptions
from datasets import CreateDataLoader
from models import create_model
from utils import util
from utils.inception_scores import inception_score


def evaluate():
    opt = EvalOptions().parse()

    res_path = Path(opt.out_dir) / "{}_{}.png".format(opt.name, opt.epoch)
    util.mkdir(opt.out_dir)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    if opt.method == "discriminator":
        scores = []
        images = []
        for data in tqdm(dataset, total=dataset_size, ascii=True):
            model.set_input(data)
            score = model.evaluate()

            images.append(model.fake.cpu())
            scores.append(score.cpu().numpy())
        scores = np.array(scores).flatten()
        score_idx = np.argsort(scores)

        num_samples = min(6*6, opt.num_samples)
        best_images = [images[i] for i in score_idx[:num_samples]]
        best_images = torch.cat(best_images, dim=0)
        best_images = best_images / 2 + 0.5
        save_image(best_images, res_path, nrow=int(math.sqrt(num_samples)))
    elif opt.method == "inception_score":
        fakes = []
        reals = []
        for data in tqdm(dataset, total=dataset_size, ascii=True):
            model.set_input(data)
            model.evaluate()

            reals.append(model.real.cpu())
            fakes.append(model.fake.cpu())

        reals = [torch.squeeze(r, dim=0) for r in reals]
        fakes = [torch.squeeze(f, dim=0) for f in fakes]

        real_mean, real_std = inception_score(reals, cuda=opt.gpu_ids, batch_size=32, resize=True, splits=1)
        print("[Real] inception Score: mean - {:.3f}, std - {:.3f}".format(real_mean, real_std))

        fake_mean, fake_std = inception_score(fakes, cuda=opt.gpu_ids, batch_size=32, resize=True, splits=1)
        print("[Fake] inception Score: mean - {:.3f}, std - {:.3f}".format(fake_mean, fake_std))


if __name__ == "__main__":
    evaluate()
