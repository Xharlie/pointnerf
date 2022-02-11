import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

import copy
import torch
import numpy as np
import time
from options import TestOptions
from data import create_data_loader, create_dataset
from models import create_model
from utils.visualizer import Visualizer
from utils import format as fmt


def main():
    torch.backends.cudnn.benchmark = True
    opt = TestOptions().parse()

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + fmt.END)

    assert opt.resume_dir is not None
    resume_dir = opt.resume_dir
    states = torch.load(os.path.join(resume_dir, '{}_states.pth'.format(opt.resume_iter)))
    epoch_count = states['epoch_count']
    total_steps = states['total_steps']
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Resume from {} epoch'.format(opt.resume_iter))
    print("Iter: ", total_steps)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    # load model
    model = create_model(opt)
    model.setup(opt)

    thres = 10
    grid, argb = model.net_ray_marching.module.build_point_cloud_visualization(0)
    mask = argb[..., 0] > thres
    points = grid[mask]
    colors = argb[mask][..., 1:4]

    import pyrender
    mesh = pyrender.Mesh.from_points(points, colors=colors)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, render_flags={'point_size': 10}, use_raymond_lighting=True)


if __name__ == '__main__':
    main()
