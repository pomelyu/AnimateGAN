from pathlib import Path
import argparse
import yaml
import torch
import arrow

import models
import datasets
from utils import util

# pylint: disable=line-too-long, W0201

def load_yaml_config(config):
    config = Path(config)
    if config.is_file:
        with config.open("r") as f:
            return yaml.load(f)

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.isTrain = True

    def initialize(self, parser):
        parser.add_argument("--config", type=str, default="", help="path to yaml config file")

        # Process
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--load_epoch", type=str, default="latest", help="which epoch to load? set to latest to use latest cached model")
        parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
        parser.add_argument("--verbose", action="store_true", help="if specified, print more debugging information")
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')

        # Environment configure
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        parser.add_argument("--num_threads", default=0, type=int, help="# threads for loading data")

        # Dataset configure
        parser.add_argument("--dataset_mode", type=str, default="anime", help="")
        parser.add_argument("--batch_size", type=int, default=8, help="input batch size")
        parser.add_argument("--load_size", type=int, default=64, help="scale images to this size")
        parser.add_argument("--crop_size", type=int, default=64, help="then crop to this size")
        parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")

        # Network configure
        parser.add_argument("--model", type=str, default="dc_gan", help="")
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt = Opt()
        parser.parse_known_args(namespace=opt)
        opt_modified = {k: getattr(opt, k) for k in opt.args() if parser.get_default(k) != getattr(opt, k)}
        opt_yaml = load_yaml_config(getattr(opt, "config"))
        opt.update_known_args(opt_yaml)
        opt.update_known_args(opt_modified)

        # modify model-related parser options
        model_name = getattr(opt, "model")
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        # parse again with the new defaults
        opt = Opt()
        parser.parse_known_args(namespace=opt)
        opt_modified = {k: getattr(opt, k) for k in opt.args() if parser.get_default(k) != getattr(opt, k)}
        opt_yaml = load_yaml_config(getattr(opt, "config"))
        opt.update_known_args(opt_yaml)
        opt.update_known_args(opt_modified)

        # modify dataset-related parser options
        dataset_name = getattr(opt, "dataset_mode")
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        opt = Opt()
        parser.parse_args(namespace=opt)
        opt_modified = {k: getattr(opt, k) for k in opt.args() if parser.get_default(k) != getattr(opt, k)}
        opt_yaml = load_yaml_config(getattr(opt, "config"))
        opt.update_args(opt_yaml)
        opt.update_args(opt_modified)

        return opt

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
        if opt.isTrain:
            expr_dir = Path(opt.checkpoints_dir) / opt.name
            util.mkdirs(expr_dir)
            local = arrow.utcnow().to("local")
            file_name = expr_dir / "opt {}.txt".format(local.format("[YYYY_MMDD] HH'mm'ss"))
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
        if isinstance(opt.gpu_ids, int):
            str_ids = [str(opt.gpu_ids)]
        else:
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


class Opt():
    def __init__(self):
        self.__raise_unknown = True

    def update_known_args(self, new_opt):
        self.__raise_unknown = False
        self._update(new_opt)

    def update_args(self, new_opt):
        self.__raise_unknown = True
        self._update(new_opt)

    def args(self):
        return [k for k in dir(self) if self._valid_key(self, k)]

    def _update(self, new_opt):
        if isinstance(new_opt, dict):
            for k, v in new_opt.items():
                if hasattr(self, k):
                    setattr(self, k, v)
                elif self.__raise_unknown:
                    raise KeyError("Unknown key: " + k)

        else:
            for k in dir(new_opt):
                if self._valid_key(new_opt, k):
                    setattr(self, k, v)

    def _valid_key(self, obj, k):
        if k.startswith("__") or callable(getattr(obj, k)):
            return False
        if not hasattr(self, k) and self.__raise_unknown:
            raise KeyError("Unknown key: " + k)
        return True
