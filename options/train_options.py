from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.is_train = True

        parser.add_argument(
            '--print_freq',
            type=int,
            default=100,
            help='frequency of showing training results on console')

        parser.add_argument('--plr',
                            type=float,
                            default=0.0005,
                            help='initial learning rate')
        parser.add_argument('--lr',
                            type=float,
                            default=0.001,
                            help='initial learning rate')
        parser.add_argument('--lr_policy',
                            type=str,
                            default='lambda',
                            help='learning rate policy: lambda|step|plateau')
        parser.add_argument(
            '--lr_decay_iters',
            type=int,
            default=50,
            help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument(
            '--lr_decay_exp',
            type=float,
            default=0.1,
            help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--train_and_test',
                            type=int,
                            default=0,
                            help='train and test at the same time')
        parser.add_argument('--test_num', type=int, default=1, help='test num')
        parser.add_argument('--test_num_step', type=int, default=1, help='test num')
        parser.add_argument('--test_freq',
                            type=int,
                            default=500,
                            help='test frequency')

        parser.add_argument('--maximum_step',
                            type=int,
                            default=None,
                            help='maximum # of training iterations')
        parser.add_argument('--niter',
                            type=int,
                            default=100,
                            help='# of iter at starting learning rate')
        parser.add_argument(
            '--niter_decay',
            type=int,
            default=100,
            help='# of iter to linearly decay learning rate to zero')


        parser.add_argument('--save_iter_freq',
                            type=int,
                            default=100000,
                            help='saving frequency')
        parser.add_argument('--prune_thresh',
                            type=float,
                            default=0.1,
                            help='saving frequency')
        parser.add_argument('--prune_iter',
                            type=int,
                            default=-1,
                            help='saving frequency')
        parser.add_argument('--prune_max_iter',
                            type=int,
                            default=9999999,
                            help='saving frequency')
        parser.add_argument('--alpha_range',
                            type=int,
                            default=0,
                            help='saving frequency')
        parser.add_argument('--prob_freq',
                            type=int,
                            default=0,
                            help='saving frequency')
        parser.add_argument('--prob_num_step',
                            type=int,
                            default=100,
                            help='saving frequency')
        parser.add_argument('--prob_mode',
                            type=int,
                            default=0,
                            help='saving frequency')
        parser.add_argument('--prob_top',
                            type=int,
                            default=1,
                            help='0 randomly select frames, 1 top frames')
        parser.add_argument('--prob_mul',
                            type=float,
                            default=1.0,
                            help='saving frequency')
        parser.add_argument('--prob_kernel_size',
                            type=float,
                            nargs='+',
                            default=None,
                            help='saving frequency')
        parser.add_argument('--prob_tiers',
                            type=int,
                            nargs='+',
                            default=(250000),
                            help='saving frequency')
        parser.add_argument('--far_thresh',
                            type=float,
                            default=-1.0,
                            help='cartisian distance for prob')
        parser.add_argument('--comb_file',
                            type=str,
                            default=None,
                            help='cartisian distance for prob')

        return parser
