import importlib
import torch.utils.data
import sys
sys.path.append("../")
from utils.ncg_string import underscore2camelcase
from .base_dataset import BaseDataset
import numpy as np

def find_dataset_class_by_name(name):
    '''
    Input
    name: string with underscore representation

    Output
    dataset: a dataset class with class name {camelcase(name)}Dataset

    Searches for a dataset module with name {name}_dataset in current
    directory, returns the class with name {camelcase(name)}Dataset found in
    the module.
    '''
    cls_name = underscore2camelcase(name) + 'Dataset'
    filename = "data.{}_dataset".format(name)
    module = importlib.import_module(filename)

    assert cls_name in module.__dict__, 'Cannot find dataset class name "{}" in "{}"'.format(
        cls_name, filename)
    cls = module.__dict__[cls_name]
    assert issubclass(cls, BaseDataset), 'Dataset class "{}" must inherit from BaseDataset'.format(cls_name)

    return cls


def get_option_setter(dataset_name):
    dataset_class = find_dataset_class_by_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    dataset = find_dataset_class_by_name(opt.dataset_name)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [{}] was created".format(instance.name()))
    return instance


def create_data_loader(opt, dataset=None):
    data_loader = DefaultDataLoader()
    data_loader.initialize(opt, dataset=dataset)
    return data_loader

def worker_init_fn(worker_id):
    # np.random.seed(np.random.get_state()[1][0] + worker_id)
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)


class DefaultDataLoader:
    def name(self):
        return self.__class__.name

    def initialize(self, opt, dataset=None):
        assert opt.batch_size >= 1
        assert opt.n_threads >= 0
        assert opt.max_dataset_size >= 1

        self.opt = opt
        self.dataset = create_dataset(opt) if dataset is None else dataset
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=not opt.serial_batches,
                                                      num_workers=int(opt.n_threads),
                                                      worker_init_fn=worker_init_fn)

    def load_data(self):
        return self.dataset

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def get_item(self, index):
        return self.dataset.get_item(index)
