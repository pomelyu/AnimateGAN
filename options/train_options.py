from .base_options import BaseOptions

# pylint: disable=line-too-long, W0201

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument("--print_freq", type=int, default=100, help="frequency of showing training results on console")
        parser.add_argument("--save_epoch_freq", type=int, default=5, help="frequency of saving checkpoints at the end of epochs")
        parser.add_argument("--save_lastest_freq", type=int, default=5000, help="frequency of saving checkpoints after iteration")

        parser.add_argument("--epoch", type=int, default=100, help="# of iter at starting learning rate")
        parser.add_argument("--epoch_decay", type=int, default=100, help="# of iter to linearly decay learning rate to zero")
        parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
        parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
        parser.add_argument("--lr_policy", type=str, default="lambda", help="learning rate policy: lambda|step|plateau|cosine")
        parser.add_argument("--lr_decay_iters", type=int, default=50, help="multiply by a gamma every lr_decay_iters iterations")

        parser.add_argument("--noise_level", type=float, default=0.0, help="max instance noise std, annealing while training")
        parser.add_argument("--flip_prob", type=float, default=0.0, help="probability of randomly flipping training label to confuse discriminator")

        parser.add_argument("--continue_train", action="store_true", help="continue training: load the latest model")
        parser.add_argument("--epoch_count", type=int, default=1, help="the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...")

        self.isTrain = True
        return parser
