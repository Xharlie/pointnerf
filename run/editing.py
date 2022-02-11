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
from options import EditOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs.mvs_points_model import MvsPointsModel
from models.mvs import mvs_utils, filter_utils

from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
from render_vid import render_vid
torch.manual_seed(0)
np.random.seed(0)
import cv2
from PIL import Image
from tqdm import tqdm



def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)




def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst



def render(model, dataset, visualizer, opt, gen_vid=False):
    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = dataset.render_total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step))
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    cam_posts = []
    cam_dirs = []
    for i in range(0, total_num):
        data = dataset.get_dummyrot_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        # cam_posts.append(data['campos'])
        # cam_dirs.append(data['raydir'] + data['campos'][None,...])
        # continue
        visuals = None
        stime = time.time()

        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            # print("tmpgts", tmpgts["gt_image"].shape)
            # print(data["pixel_idx"])
            model.set_input(data)
            model.test()
            curr_visuals = model.get_current_visuals(data=data)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    visuals[key][start:end, :] = value.cpu().numpy()

        for key, value in visuals.items():
            visualizer.print_details("{}:{}".format(key, visuals[key].shape))
            visuals[key] = visuals[key].reshape(height, width, 3)
        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i)

    # visualizer.save_neural_points(200, np.concatenate(cam_posts, axis=0),None, None, save_ref=False)
    # visualizer.save_neural_points(200, np.concatenate(cam_dirs, axis=0),None, None, save_ref=False)
    # print("vis")
    # exit()

    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num), 0)
        print('--------------------------------Finish generating vid--------------------------------')

    return



def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]




def load_parts_info(opt, name, inds_name, trans_name):
    resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(os.path.join(opt.checkpoints_dir , "edit_srcs" , name))
    checkpoint = os.path.join(opt.checkpoints_dir , "edit_srcs" , name , "{}_net_ray_marching.pth".format(resume_iter))
    trans_file = None if trans_name.strip() == "no" else os.path.join(opt.checkpoints_dir , "edit_srcs" , name , "transforms", trans_name + ".txt")
    inds_file = None if inds_name.strip() == "all" else os.path.join(opt.checkpoints_dir , "edit_srcs" , name , "parts_index", inds_name + ".txt")
    Matrix = torch.eye(4, device="cuda", dtype=torch.float32) if trans_file is None else np.loadtxt(trans_file)
    Rot = Matrix[:3,:3]
    Translation = Matrix[:3, 3]
    saved_features = torch.load(checkpoint, map_location="cuda")
    print("loaded neural points from ", checkpoint, saved_features.keys())
    if inds_file is None:
        inds = torch.ones(len(saved_features["neural_points.xyz"]), dtype=torch.bool, device="cuda")
    else:
        inds = np.loadtxt(inds_file)
    return saved_features, inds, Rot, Translation



def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def main():
    torch.backends.cudnn.benchmark = True
    opt = EditOptions().parse()
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Debug Mode')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    visualizer = Visualizer(opt)

    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(32, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.split = "test"
    # test_dataset = create_dataset(test_opt)
    img_lst=None

    opt.is_train = False
    opt.mode = 2
    if opt.resume_iter == "best":
        opt.resume_iter = "latest"
    opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)

    model = create_model(opt)

    points_xyz_all = torch.zeros([0,3], device="cuda", dtype=torch.float32)
    points_embedding_all = torch.zeros([1,0,63], device="cuda", dtype=torch.float32)
    points_conf_all = torch.zeros([1,0,1], device="cuda", dtype=torch.float32)
    points_color_all = torch.zeros([1,0,3], device="cuda", dtype=torch.float32)
    points_dir_all = torch.zeros([1,0,3], device="cuda", dtype=torch.float32)
    Rw2c_all = torch.zeros([0,3,3], device="cuda", dtype=torch.float32)


    for name, inds_name, trans_name in zip(opt.neural_points_names, opt.parts_index_names, opt.Transformation_names):
        saved_features, inds, Rot, Tran = load_parts_info(opt, name, inds_name, trans_name)
        inds = torch.as_tensor(inds, dtype=torch.bool, device="cuda")
        Rot = torch.as_tensor(Rot, dtype=torch.float32, device=inds.device)
        Tran = torch.as_tensor(Tran, dtype=torch.float32, device=inds.device)
        xyz, points_embeding, points_conf, points_dir, points_color, eulers, Rw2c = saved_features["neural_points.xyz"][inds,:], saved_features["neural_points.points_embeding"][:,inds,:] if "neural_points.points_embeding" in saved_features else None,saved_features["neural_points.points_conf"][:,inds,:] if "neural_points.points_conf" in saved_features else None, saved_features["neural_points.points_dir"][:,inds,:] if "neural_points.points_dir" in saved_features else None, saved_features["neural_points.points_color"][:,inds,:] if "neural_points.points_color" in saved_features else None, saved_features["neural_points.eulers"] if "neural_points.eulers" in saved_features else None, saved_features["neural_points.Rw2c"] if "neural_points.Rw2c" in saved_features else None

        Mat = torch.eye(4, device=Rot.device, dtype=torch.float32)
        Mat[:3,:3] = Rot
        Mat[:3,3] = Tran
        xyz = (torch.cat([xyz, torch.ones_like(xyz[:,:1])], dim=-1) @ Mat.transpose(0,1))[:,:3]
        print("Rot", Rot)
        Rw2c = Rot if Rw2c is None else Rw2c @ Rot.transpose(0,1) #.transpose(0,1) # w2c is reversed against movement
        Rw2c = Rw2c[None, ...].expand(len(xyz),-1,-1)

        points_xyz_all = torch.cat([points_xyz_all, xyz], dim=0)
        Rw2c_all = torch.cat([Rw2c_all, Rw2c], dim=0)
        points_embedding_all = torch.cat([points_embedding_all, points_embeding], dim=1)
        points_conf_all = torch.cat([points_conf_all, points_conf], dim=1)
        points_color_all = torch.cat([points_color_all, points_color], dim=1)
        points_dir_all = torch.cat([points_dir_all, points_dir], dim=1)

    model.set_points(points_xyz_all.cuda(), points_embedding_all.cuda(), points_color=points_color_all.cuda(),
                     points_dir=points_dir_all.cuda(), points_conf=points_conf_all.cuda(), Rw2c=Rw2c_all.cuda(), editing=True)

    visualizer.save_neural_points("pnts", model.neural_points.xyz, None, None, save_ref=False)
    print("vis")
    # exit()

    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.test_num_step=1 # opt.test_num_step
    test_opt.name = opt.name + "/{}".format(opt.render_name)
    test_opt.render_only = 1
    model.opt.no_loss = 1
    model.opt.is_train = 0
    model.setup(opt)

    print("full datasets test:")
    test_dataset = create_dataset(test_opt)
    render(model, test_dataset, Visualizer(test_opt), test_opt, gen_vid=True)
    # model.opt.no_loss = 0
    # model.opt.is_train = 1
    other_states = {
        'epoch_count': 0,
        'total_steps': 0,
    }
    print('saving model ({}, epoch {}, total_steps {})'.format(opt.name, 0, 0))
    model.save_networks(0, other_states)

#
# def save_points_conf(visualizer, xyz, points_color, points_conf, total_steps):
#     print("total:", xyz.shape, points_color.shape, points_conf.shape)
#     colors, confs = points_color[0], points_conf[0,...,0]
#     pre = -1000
#     for i in range(12):
#         thresh = (i * 0.1) if i <= 10 else 1000
#         mask = ((confs <= thresh) * (confs > pre)) > 0
#         thresh_xyz = xyz[mask, :]
#         thresh_color = colors[mask, :]
#         visualizer.save_neural_points(f"{total_steps}-{thresh}", thresh_xyz, thresh_color[None, ...], None, save_ref=False)
#         pre = thresh
#     exit()




if __name__ == '__main__':
    main()
