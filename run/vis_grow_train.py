import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs.mvs_points_model import MvsPointsModel
from models.mvs import mvs_utils, filter_utils

from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
# from render_vid import render_vid
torch.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
import cv2
from PIL import Image
import imageio
from utils.util import to8b


def read_image(filepath, dtype=None):
    image = np.asarray(Image.open(filepath))
    if dtype is not None and dtype == np.float32:
        image = (image / 255).astype(dtype)
    return image


def render_grow(pnt_dir, iters, vids):
    print('-----------------------------------Rendering Grow-----------------------------------')


    # visualizer.save_neural_points(200, np.concatenate(cam_posts, axis=0),None, None, save_ref=False)
    # visualizer.save_neural_points(200, np.concatenate(cam_dirs, axis=0),None, None, save_ref=False)
    # print("vis")
    # exit()
    for t in tqdm(range(len(vids))):
        vid = vids[t]
        img_lst = []
        for iter in iters:
            img_dir = os.path.join(pnt_dir, 'prob_img_{}'.format(iter))
            #  ''step-{:04d}-{}.png'.format(i, name))
            img_filepath = os.path.join(img_dir, "step-{}-0-ref0.png".format(vid))
            img_arry = read_image(img_filepath, dtype=np.float32)
            img_lst.append(img_arry)

            stacked_imgs = [to8b(img_arry) for img_arry in img_lst]
            filename = 'grow_video_{:04d}.mov'.format(vid)
            imageio.mimwrite(os.path.join(pnt_dir, filename), stacked_imgs, fps=3, quality=8)
            filename = 'grow_video_{:04d}.gif'.format(vid)
            imageio.mimwrite(os.path.join(pnt_dir, filename), stacked_imgs, fps=3, format='GIF')
    return


if __name__ == '__main__':
    pnt_dir = "/home/xharlie/user_space/codes/testNr/checkpoints/scan103_normcam2_confcolordir_KNN8_LRelu_grid800_dmsk_full2geo0_agg2_zeroone1e4_confree_prl2e3_probe2e3_1_comb/points"
    iters = list(range(1000, 25000, 1000))
    vids = list(range(16, 20))
    render_grow(pnt_dir, iters, vids)
