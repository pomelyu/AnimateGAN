from pathlib import Path
import time
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from .util import mkdir

class Visualizer():
    def __init__(self, opt):
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        mkdir(expr_dir)
        self.writer = SummaryWriter(log_dir=str(expr_dir))
        self.log_file = expr_dir / "log.txt"
        with self.log_file.open("a") as f:
            now = time.strftime("%c")
            f.write("================ Training Loss ({}) ================\n".format(now))

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)

    def add_image(self, name, image, n_iter, grid_rows=8):
        grid = vutils.make_grid(image, normalize=True, scale_each=True, range=(-1, 1), nrow=grid_rows)
        self.writer.add_image(name, grid, n_iter)

    def add_log(self, msg):
        print(msg)
        with self.log_file.open("a") as f:
            f.write("\n{}".format(msg))
