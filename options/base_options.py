from pathlib import Path
import argparse
import torch

import models
import datasets
from utils import util

# pylint: disable=line-too-long, W0201

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.isTrain = True

    def initialize(self, parser):
        parser.add_argument("--dataroot", required=True, help="path to images")
        parser.add_argument("--load_size", type=int, default=64, help="scale images to this size")
        parser.add_argument("--crop_size", type=int, default=64, help="then crop to this size")
        parser.add_argument("--batch_size", type=int, default=8, help="input batch size")
        parser.add_argument("--dataset_mode", type=str, default="anime", help="")
        parser.add_argument("--num_threads", default=4, type=int, help="# threads for loading data")
        parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")

        parser.add_argument("--model", type=str, default="dc_gan", help="")
        parser.add_argument("--latent_size", type=int, default=100, help="the dimension of latent")
        parser.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer")
        parser.add_argument("--ndf", type=int, default=64, help="# of discrim filters in first conv layer")

        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")

        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = Path(opt.checkpoints_dir) / opt.name
        util.mkdirs(expr_dir)
        file_name = expr_dir / "opt.txt"
        with file_name.open(mode="w") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(",")
        opt.gpu_ids = []
        for str_id in str_ids:
            gpu_id = int(str_id)
            if gpu_id >= 0:
                opt.gpu_ids.append(gpu_id)
        if opt.gpu_ids:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
