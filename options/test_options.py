from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        self.is_train = False
        parser.add_argument('--test_num', type=int, default=1, help='test num')
        parser.add_argument('--test_num_step', type=int, default=1, help='test num')
        parser.add_argument('--test_printId', type=int, default=0, help='The first id (offset) when saving.')
        return parser
