from pathlib import Path
from tqdm import tqdm
import imageio

from options.eval_options import EvalOptions
from datasets import CreateDataLoader
from models import create_model
from utils import util

def test_distangle():
    opt = EvalOptions().parse()

    out_dir = Path(opt.out_dir)
    util.mkdir(out_dir)

    dataloader = CreateDataLoader(opt)
    dataset = dataloader.load_data()
    dataset_size = len(dataloader)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    image_size = opt.crop_size
    for data in tqdm(dataset, total=dataset_size, ascii=True):
        model.set_input(data)
        model.forward()

        res = model.get_test_output()
        path = model.get_image_paths()[0]
        file_name = Path(path).with_suffix(".gif").name

        frames = [res[of:of+image_size] for of in range(0, res.shape[0], image_size)]
        imageio.mimsave(out_dir / file_name, frames, 'GIF', duration=0.05)

        # imageio.imwrite(, res)

if __name__ == "__main__":
    test_distangle()
