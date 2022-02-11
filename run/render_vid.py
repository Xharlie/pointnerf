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

from tqdm import trange


def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def render_vid(model, dataset, visualizer, opt, total_steps):

    print(
        '-----------------------------------Rendering Vid-----------------------------------'
    )
    model.eval()
    render_num = len(dataset.render_poses)
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visual_lst = []
    for i in range(render_num):
        data = dataset.get_dummyrot_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        visuals = None
        starttime=time.time()
        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])

            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            # data['gt_image'] = gt_image[:, start:end, :]
            # data['gt_mask'] = gt_mask[:, start:end, :]
            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals()

            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None or value.shape[-1] != 3 or not key.endswith("color"):
                        continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if value is None or value.shape[-1] != 3 or not key.endswith("color"):
                        continue
                    visuals[key][start:end, :] = value.cpu().numpy()
        for key, value in visuals.items():
            visuals[key] = visuals[key].reshape(height, width, 3)
        visual_lst.append(visuals)
        print("render time:", time.time() - starttime)
    visualizer.display_video(visual_lst, total_steps)
    model.train()
    print(
        '--------------------------------Finish Rendering--------------------------------'
    )
    return


def main():
    torch.backends.cudnn.benchmark = True
    opt = TestOptions().parse()
    opt.no_loss = True
    opt.gpu_ids='0'

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED + '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' + fmt.END)

    if opt.resume_dir:
        resume_dir = opt.resume_dir
        resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
        opt.resume_iter = resume_iter
        states = torch.load(os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)))
        epoch_count = states['epoch_count']
        total_steps = states['total_steps']
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Test {} at epoch {}'.format(opt.resume_dir, opt.resume_iter))
        print("Iter: ", total_steps)
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        epoch_count = 1
        total_steps = 0

    # load model
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_dataset = create_dataset(test_opt)
    dataset_size = len(test_dataset)

    print('# training images = {}'.format(dataset_size))

    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    render_vid(model, test_dataset, visualizer, test_opt, total_steps)


if __name__ == '__main__':
    main()
