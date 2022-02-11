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
from run.evaluate import report_metrics
torch.manual_seed(0)
np.random.seed(0)


def test(model, dataset, visualizer, opt, test_steps=0):

    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = dataset.total
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    for i in range(min(opt.test_num,total_num)):
        data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        tmpgts={}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone()
        
        data.pop('gt_image', None)
        data.pop('gt_mask', None)
        
        visuals = None
        stime = time.time()
        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])

            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals()
            
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None:
                        continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if value is None:
                        continue
                    visuals[key][start:end, :] = value.cpu().numpy()
        visuals['gt_image'] = tmpgts['gt_image']
        if 'gt_mask' in curr_visuals:
            visuals['gt_mask'] = np.zeros((height * width, 3)).astype(chunk.dtype)
            visuals['gt_mask'][:] = tmpgts['gt_mask']
        for key, value in visuals.items():
            print(key, visuals[key].shape)
            visuals[key] = visuals[key].reshape(height, width, 3)
        print("time used: {} s".format(time.time()-stime))
        visualizer.display_current_results(visuals, opt.test_printId+i)

    print('--------------------------------Finish Test Rendering--------------------------------')

    report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr", "ssim", "rmse"], [i for i in range(opt.test_printId, opt.test_printId+min(opt.test_num,total_num))], imgStr="step-%04d-{}_raycolor.png".format("coarse" if opt.fine_sample_num==0 else "fine"))

    print('--------------------------------Finish Evaluation--------------------------------')
    return

def get_latest_epoch(resume_dir):
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]

def main():
    torch.backends.cudnn.benchmark = True
    opt = TestOptions().parse()
    opt.no_loss = True
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

    test(model, test_dataset, visualizer, test_opt, total_steps)
    

if __name__ == '__main__':
    main()
