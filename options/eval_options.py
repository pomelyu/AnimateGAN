from .base_options import BaseOptions

# pylint: disable=line-too-long, W0201

class EvalOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--out_dir', type=str, default="out", help='the output folder path')
        parser.add_argument("--method", type=str, default="discriminator", help="how to evaluate network, ex: discriminator, inception_score")
        parser.add_argument("--num_samples", type=int, default=400, help="number of samples to evaluate")

        parser.set_defaults(batch_size=1)
        self.isTrain = False
        return parser
