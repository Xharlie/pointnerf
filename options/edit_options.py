from .base_options import BaseOptions

class EditOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.is_train = False
        parser.add_argument('--neural_points_names',
                            type=str,
                            nargs='+',
                            # default=["imgfeat_0_0123", "vol"],
                            default=["imgfeat_0_0", "vol"],
                            help="which feature_map")
        parser.add_argument('--Transformation_names',
                            type=str,
                            nargs='+',
                            # default=["imgfeat_0_0123", "vol"],
                            default=["1", "2"],
                            help="which feature_map")
        parser.add_argument('--render_name',
                            type=str,
                            # default=["imgfeat_0_0123", "vol"],
                            default="tryout",
                            help="which feature_map")
        parser.add_argument('--parts_index_names',
                            type=str,
                            nargs='+',
                            # default=["imgfeat_0_0123", "vol"],
                            default=["1", "2"],
                            help="which feature_map")
        parser.add_argument('--render_stride',
                            type=int,
                            default=30,
                            help='feed batches in order without shuffling')
        parser.add_argument('--render_radius',
                            type=float,
                            default=4.0,
                            help='feed batches in order without shuffling')
        return parser
