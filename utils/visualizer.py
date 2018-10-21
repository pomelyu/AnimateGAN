from pathlib import Path
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from .util import mkdir

class Visualizer():
    def __init__(self, opt):
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        mkdir(expr_dir)
        self.writer = SummaryWriter(log_dir=str(expr_dir))

    def add_scalar(self, name, value, n_iter):
        self.writer.add_scalar(name, value, n_iter)

    def add_image(self, name, image, n_iter, grid_rows=8):
        grid = vutils.make_grid(image, normalize=True, scale_each=True, range=(-1, 1), nrow=grid_rows)
        self.writer.add_image(name, grid, n_iter)
