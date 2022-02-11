import torch.utils.data as data
from PIL import Image


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return self.__class__.__name__

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
