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
from pprint import pprint
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
# from render_vid import render_vid
torch.manual_seed(0)
np.random.seed(0)
import random
import cv2
from PIL import Image
from tqdm import tqdm
# from models.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
# from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
import gc

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)


def nearest_view(campos, raydir, xyz, id_list):
    cam_ind = torch.zeros([0,1], device=campos.device, dtype=torch.long)
    step=10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz),i+step), None, :] - campos[None, ...] # N, M, 3
        dists_norm = torch.norm(dists, dim=-1) # N, M
        dists_dir = dists / (dists_norm[...,None]+1e-6) # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :],dim=-1)) # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1,1)], dim=0) # N, 1
    return cam_ind


def gen_points_filter_embeddings(dataset, visualizer, opt):
    print('-----------------------------------Generate Points-----------------------------------')
    opt.is_train=False
    opt.mode = 1
    model = create_model(opt)
    model.setup(opt)

    model.eval()
    cam_xyz_all = []
    intrinsics_all = []
    extrinsics_all = []
    confidence_all = []
    points_mask_all = []
    intrinsics_full_lst = []
    confidence_filtered_all = []
    near_fars_all = []
    gpu_filter = True
    cpu2gpu= len(dataset.view_id_list) > 300

    imgs_lst, HDWD_lst, c2ws_lst, w2cs_lst, intrinsics_lst = [],[],[],[],[]
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset.view_id_list))):
            data = dataset.get_init_item(i)
            model.set_input(data)
            # intrinsics    1, 3, 3, 3
            points_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, c2ws, w2cs, intrinsics, near_fars  = model.gen_points()
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            B, N, C, H, W, _ = points_xyz_lst[0].shape
            # print("points_xyz_lst",points_xyz_lst[0].shape)
            cam_xyz_all.append((points_xyz_lst[0].cpu() if cpu2gpu else points_xyz_lst[0]) if gpu_filter else points_xyz_lst[0].cpu().numpy())
            # intrinsics_lst[0] 1, 3, 3
            intrinsics_all.append(intrinsics_lst[0] if gpu_filter else intrinsics_lst[0])
            extrinsics_all.append(extrinsics_lst[0] if gpu_filter else extrinsics_lst[0].cpu().numpy())
            if opt.manual_depth_view !=0:
                confidence_all.append((photometric_confidence_lst[0].cpu() if cpu2gpu else photometric_confidence_lst[0]) if gpu_filter else photometric_confidence_lst[0].cpu().numpy())
            points_mask_all.append((point_mask_lst[0].cpu() if cpu2gpu else point_mask_lst[0]) if gpu_filter else point_mask_lst[0].cpu().numpy())
            imgs_lst.append(data["images"].cpu())
            HDWD_lst.append(HDWD)
            c2ws_lst.append(c2ws)
            w2cs_lst.append(w2cs)
            intrinsics_full_lst.append(intrinsics)
            near_fars_all.append(near_fars[0,0])
            # visualizer.save_neural_points(i, points_xyz_lst[0], None, data, save_ref=opt.load_points == 0)
            # #################### start query embedding ##################
        torch.cuda.empty_cache()
        if opt.manual_depth_view != 0:
            if gpu_filter:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=True, return_w=True, cpu2gpu=cpu2gpu, near_fars_all=near_fars_all)
            else:
                _, xyz_world_all, confidence_filtered_all = filter_utils.filter_by_masks(cam_xyz_all, [intr.cpu().numpy() for intr in intrinsics_all], extrinsics_all, confidence_all, points_mask_all, opt)
            # print(xyz_ref_lst[0].shape) # 224909, 3
        else:
            cam_xyz_all = [cam_xyz_all[i].reshape(-1,3)[points_mask_all[i].reshape(-1),:] for i in range(len(cam_xyz_all))]
            xyz_world_all = [np.matmul(np.concatenate([cam_xyz_all[i], np.ones_like(cam_xyz_all[i][..., 0:1])], axis=-1), np.transpose(np.linalg.inv(extrinsics_all[i][0,...])))[:, :3] for i in range(len(cam_xyz_all))]
            xyz_world_all, cam_xyz_all, confidence_filtered_all = filter_by_masks.range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_all, opt)
            del cam_xyz_all
        # for i in range(len(xyz_world_all)):
        #     visualizer.save_neural_points(i, torch.as_tensor(xyz_world_all[i], device="cuda", dtype=torch.float32), None, data, save_ref=opt.load_points==0)
        # exit()
        # xyz_world_all = xyz_world_all.cuda()
        # confidence_filtered_all = confidence_filtered_all.cuda()
        points_vid = torch.cat([torch.ones_like(xyz_world_all[i][...,0:1]) * i for i in range(len(xyz_world_all))], dim=0)
        xyz_world_all = torch.cat(xyz_world_all, dim=0) if gpu_filter else torch.as_tensor(
            np.concatenate(xyz_world_all, axis=0), device="cuda", dtype=torch.float32)
        confidence_filtered_all = torch.cat(confidence_filtered_all, dim=0) if gpu_filter else torch.as_tensor(np.concatenate(confidence_filtered_all, axis=0), device="cuda", dtype=torch.float32)
        print("xyz_world_all", xyz_world_all.shape, points_vid.shape, confidence_filtered_all.shape)
        torch.cuda.empty_cache()
        # visualizer.save_neural_points(0, xyz_world_all, None, None, save_ref=False)
        # print("vis 0")

        print("%%%%%%%%%%%%%  getattr(dataset, spacemin, None)", getattr(dataset, "spacemin", None))
        if getattr(dataset, "spacemin", None) is not None:
            mask = (xyz_world_all - dataset.spacemin[None, ...].to(xyz_world_all.device)) >= 0
            mask *= (dataset.spacemax[None, ...].to(xyz_world_all.device) - xyz_world_all) >= 0
            mask = torch.prod(mask, dim=-1) > 0
            first_lst, second_lst = masking(mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
        # visualizer.save_neural_points(50, xyz_world_all, None, None, save_ref=False)
        # print("vis 50")
        if getattr(dataset, "alphas", None) is not None:
            vishull_mask = mvs_utils.alpha_masking(xyz_world_all, dataset.alphas, dataset.intrinsics, dataset.cam2worlds, dataset.world2cams, dataset.near_far if opt.ranges[0] < -90.0 and getattr(dataset,"spacemin",None) is None else None, opt=opt)
            first_lst, second_lst = masking(vishull_mask, [xyz_world_all, points_vid, confidence_filtered_all], [])
            xyz_world_all, points_vid, confidence_filtered_all = first_lst
            print("alpha masking xyz_world_all", xyz_world_all.shape, points_vid.shape)
        # visualizer.save_neural_points(100, xyz_world_all, None, data, save_ref=opt.load_points == 0)
        # print("vis 100")

        if opt.vox_res > 0:
            xyz_world_all, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all)//99999999+1),...].cuda(), opt.vox_res)
            points_vid = points_vid[sampled_pnt_idx,:]
            confidence_filtered_all = confidence_filtered_all[sampled_pnt_idx]
            print("after voxelize:", xyz_world_all.shape, points_vid.shape)
            xyz_world_all = xyz_world_all.cuda()

        xyz_world_all = [xyz_world_all[points_vid[:,0]==i, :] for i in range(len(HDWD_lst))]
        confidence_filtered_all = [confidence_filtered_all[points_vid[:,0]==i] for i in range(len(HDWD_lst))]
        cam_xyz_all = [(torch.cat([xyz_world_all[i], torch.ones_like(xyz_world_all[i][...,0:1])], dim=-1) @ extrinsics_all[i][0].t())[...,:3] for i in range(len(HDWD_lst))]
        points_embedding_all, points_color_all, points_dir_all, points_conf_all = [], [], [], []
        for i in tqdm(range(len(HDWD_lst))):
            if len(xyz_world_all[i]) > 0:
                embedding, color, dir, conf = model.query_embedding(HDWD_lst[i], torch.as_tensor(cam_xyz_all[i][None, ...], device="cuda", dtype=torch.float32), torch.as_tensor(confidence_filtered_all[i][None, :, None], device="cuda", dtype=torch.float32) if len(confidence_filtered_all) > 0 else None, imgs_lst[i].cuda(), c2ws_lst[i], w2cs_lst[i], intrinsics_full_lst[i], 0, pointdir_w=True)
                points_embedding_all.append(embedding)
                points_color_all.append(color)
                points_dir_all.append(dir)
                points_conf_all.append(conf)

        xyz_world_all = torch.cat(xyz_world_all, dim=0)
        points_embedding_all = torch.cat(points_embedding_all, dim=1)
        points_color_all = torch.cat(points_color_all, dim=1) if points_color_all[0] is not None else None
        points_dir_all = torch.cat(points_dir_all, dim=1) if points_dir_all[0] is not None else None
        points_conf_all = torch.cat(points_conf_all, dim=1) if points_conf_all[0] is not None else None

        visualizer.save_neural_points(200, xyz_world_all, points_color_all, data, save_ref=opt.load_points == 0)
        print("vis")
        model.cleanup()
        del model
    return xyz_world_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, [img[0].cpu() for img in imgs_lst], [c2w for c2w in c2ws_lst], [w2c for w2c in w2cs_lst] , intrinsics_all, [list(HDWD) for HDWD in HDWD_lst]


def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst



def render_vid(model, dataset, visualizer, opt, bg_info, steps=0, gen_vid=True):
    print('-----------------------------------Rendering-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step))
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
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
            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

            model.test()
            curr_visuals = model.get_current_visuals(data=data)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
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



def test(model, dataset, visualizer, opt, bg_info, test_steps=0, gen_vid=False, lpips=True):
    print('-----------------------------------Testing-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step)) # 1 if test_steps == 10000 else opt.test_num_step
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    count = 0;
    for i in range(0, total_num, opt.test_num_step): # 1 if test_steps == 10000 else opt.test_num_step
        data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        edge_mask = torch.zeros([height, width], dtype=torch.bool)
        edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
        edge_mask=edge_mask.reshape(-1) > 0
        np_edge_mask=edge_mask.numpy().astype(bool)
        totalpixel = pixel_idx.shape[1]
        tmpgts = {}
        tmpgts["gt_image"] = data['gt_image'].clone()
        tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None

        # data.pop('gt_image', None)
        data.pop('gt_mask', None)

        visuals = None
        stime = time.time()
        ray_masks = []
        for k in range(0, totalpixel, chunk_size):
            start = k
            end = min([k + chunk_size, totalpixel])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            model.set_input(data)

            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

                # xyz_world_sect_plane_lst.append(xyz_world_sect_plane)
            model.test()
            curr_visuals = model.get_current_visuals(data=data)

            # print("loss", mse2psnr(torch.nn.MSELoss().to("cuda")(curr_visuals['coarse_raycolor'], tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :].cuda())))
            # print("sum", torch.sum(torch.square(tmpgts["gt_image"].view(1, -1, 3)[:, start:end, :] - tmpgts["gt_image"].view(height, width, 3)[data["pixel_idx"][0,...,1].long(), data["pixel_idx"][0,...,0].long(),:])))
            chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height, width, 3)).astype(chunk.dtype)
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if value is None or key=="gt_image":
                        continue
                    visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
            if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                ray_masks.append(model.output["ray_mask"] > 0)
        if len(ray_masks) > 0:
            ray_masks = torch.cat(ray_masks, dim=1)
        # visualizer.save_neural_points(data["id"].cpu().numpy()[0], (raydir.cuda() + data["campos"][:, None, :]).squeeze(0), None, data, save_ref=True)
        # exit()
        # print("curr_visuals",curr_visuals)
        pixel_idx=pixel_idx.to(torch.long)
        gt_image = torch.zeros((height*width, 3), dtype=torch.float32)
        gt_image[edge_mask, :] = tmpgts['gt_image'].clone()
        if 'gt_image' in model.visual_names:
            visuals['gt_image'] = gt_image
        if 'gt_mask' in curr_visuals:
            visuals['gt_mask'] = np.zeros((height, width, 3)).astype(chunk.dtype)
            visuals['gt_mask'][np_edge_mask,:] = tmpgts['gt_mask']
        if 'ray_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
            visuals['ray_masked_coarse_raycolor'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
            visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
            visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'ray_depth_masked_gt_image' in model.visual_names:
            visuals['ray_depth_masked_gt_image'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['ray_depth_masked_gt_image'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
        if 'gt_image_ray_masked' in model.visual_names:
            visuals['gt_image_ray_masked'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
            visuals['gt_image_ray_masked'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
        for key, value in visuals.items():
            if key in opt.visual_items:
                visualizer.print_details("{}:{}".format(key, visuals[key].shape))
                visuals[key] = visuals[key].reshape(height, width, 3)


        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i, opt=opt)

        acc_dict = {}
        if "coarse_raycolor" in opt.test_color_loss_items:
            loss = torch.nn.MSELoss().to("cuda")(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), gt_image.view(1, -1, 3).cuda())
            acc_dict.update({"coarse_raycolor": loss})
            print("coarse_raycolor", loss, mse2psnr(loss))

        if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
            masked_gt = tmpgts["gt_image"].view(1, -1, 3).cuda()[ray_masks,:].reshape(1, -1, 3)
            ray_masked_coarse_raycolor = torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3)[:,edge_mask,:][ray_masks,:].reshape(1, -1, 3)

            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_gt")
            # filepath = os.path.join("/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # tmpgtssave = tmpgts["gt_image"].view(1, -1, 3).clone()
            # tmpgtssave[~ray_masks,:] = 1.0
            # img = np.array(tmpgtssave.view(height,width,3))
            # save_image(img, filepath)
            #
            # filename = 'step-{:04d}-{}-vali.png'.format(i, "masked_coarse_raycolor")
            # filepath = os.path.join(
            #     "/home/xharlie/user_space/codes/testNr/checkpoints/fdtu_try/test_{}/images".format(38), filename)
            # csave = torch.zeros_like(tmpgts["gt_image"].view(1, -1, 3))
            # csave[~ray_masks, :] = 1.0
            # csave[ray_masks, :] = torch.as_tensor(visuals["coarse_raycolor"]).view(1, -1, 3)[ray_masks,:]
            # img = np.array(csave.view(height, width, 3))
            # save_image(img, filepath)

            loss = torch.nn.MSELoss().to("cuda")(ray_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_masked_coarse_raycolor", loss, mse2psnr(loss)))

        if "ray_depth_mask" in model.output and "ray_depth_masked_coarse_raycolor" in opt.test_color_loss_items:
            ray_depth_masks = model.output["ray_depth_mask"].reshape(model.output["ray_depth_mask"].shape[0], -1)
            masked_gt = torch.masked_select(tmpgts["gt_image"].view(1, -1, 3).cuda(), (ray_depth_masks[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
            ray_depth_masked_coarse_raycolor = torch.masked_select(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), ray_depth_masks[..., None].expand(-1, -1, 3).reshape(1, -1, 3))

            loss = torch.nn.MSELoss().to("cuda")(ray_depth_masked_coarse_raycolor, masked_gt)
            acc_dict.update({"ray_depth_masked_coarse_raycolor": loss})
            visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_depth_masked_coarse_raycolor", loss, mse2psnr(loss)))
        print(acc_dict.items())
        visualizer.accumulate_losses(acc_dict)
        count+=1

    visualizer.print_losses(count)
    psnr = visualizer.get_psnr(opt.test_color_loss_items[0])
    # visualizer.reset()

    print('--------------------------------Finish Test Rendering--------------------------------')

    report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr", "ssim", "lpips", "vgglpips", "rmse"] if lpips else ["psnr", "ssim", "rmse"], [i for i in range(0, total_num, opt.test_num_step)], imgStr="step-%04d-{}.png".format(opt.visual_items[0]),gtStr="step-%04d-{}.png".format(opt.visual_items[1]))


    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num, opt.test_num_step), test_steps)
        print('--------------------------------Finish generating vid--------------------------------')
    return psnr


def probe_hole(model, dataset, visualizer, opt, bg_info, test_steps=0, opacity_thresh=0.7):
    print('-----------------------------------Probing Holes-----------------------------------')
    add_xyz = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_conf = torch.zeros([0, 1], device="cuda", dtype=torch.float32)
    add_color = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_dir = torch.zeros([0, 3], device="cuda", dtype=torch.float32)
    add_embedding = torch.zeros([0, opt.point_features_dim], device="cuda", dtype=torch.float32)
    kernel_size = model.opt.kernel_size
    if opt.prob_kernel_size is not None:
        tier = np.sum(np.asarray(opt.prob_tiers) < test_steps)
        print("cal by tier", tier)
        model.opt.query_size = np.asarray(opt.prob_kernel_size[tier*3:tier*3+3])
        print("prob query size =", model.opt.query_size)
    model.opt.prob = 1
    total_num = len(model.top_ray_miss_ids) -1 if opt.prob_mode == 0 and opt.prob_num_step > 1 else len(dataset)

    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size
    height = dataset.height
    width = dataset.width
    visualizer.reset()

    max_num = len(dataset) // opt.prob_num_step
    take_top = False
    if opt.prob_top == 1 and opt.prob_mode <= 0: # and opt.far_thresh <= 0:
        if getattr(model, "top_ray_miss_ids", None) is not None:
            mask = model.top_ray_miss_loss[:-1] > 0.0
            frame_ids = model.top_ray_miss_ids[:-1][mask][:max_num]
            print(len(frame_ids), max_num)
            print("prob frame top_ray_miss_loss:", model.top_ray_miss_loss)
            take_top = True
        else:
            print("model has no top_ray_miss_ids")
    else:
        frame_ids = list(range(len(dataset)))[:max_num]
        random.shuffle(frame_ids)
        frame_ids = frame_ids[:max_num]
    print("{}/{} has holes, id_lst to prune".format(len(frame_ids), total_num), frame_ids, opt.prob_num_step)
    print("take top:", take_top, "; prob frame ids:", frame_ids)
    with tqdm(range(len(frame_ids))) as pbar:
        for j in pbar:
            i = frame_ids[j]
            pbar.set_description("Processing frame id %d" % i)
            data = dataset.get_item(i)
            bg = data['bg_color'][None, :].cuda()
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([height, width], dtype=torch.bool, device='cuda')
            edge_mask[pixel_idx[0, ..., 1].to(torch.long), pixel_idx[0, ..., 0].to(torch.long)] = 1
            edge_mask = edge_mask.reshape(-1) > 0
            totalpixel = pixel_idx.shape[1]
            gt_image_full = data['gt_image'].cuda()

            probe_keys = ["coarse_raycolor", "ray_mask", "ray_max_sample_loc_w", "ray_max_far_dist", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", "shading_avg_embedding"]
            prob_maps = {}
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)
                output = model.test()
                chunk_pixel_id = data["pixel_idx"].to(torch.long)
                output["ray_mask"] = output["ray_mask"][..., None]

                for key in probe_keys:
                    if "ray_max_shading_opacity" not in output and key != 'coarse_raycolor':
                        break
                    if output[key] is None:
                        prob_maps[key] = None
                    else:
                        if key not in prob_maps.keys():
                            C = output[key].shape[-1]
                            prob_maps[key] = torch.zeros((height, width, C), device="cuda", dtype=output[key].dtype)
                        prob_maps[key][chunk_pixel_id[0, ..., 1], chunk_pixel_id[0, ..., 0], :] = output[key]

            gt_image = torch.zeros((height * width, 3), dtype=torch.float32, device=prob_maps["ray_mask"].device)
            gt_image[edge_mask, :] = gt_image_full
            gt_image = gt_image.reshape(height, width, 3)
            miss_ray_mask = (prob_maps["ray_mask"] < 1) * (torch.norm(gt_image - bg, dim=-1, keepdim=True) > 0.002)
            miss_ray_inds = (edge_mask.reshape(height, width, 1) * miss_ray_mask).squeeze(-1).nonzero() # N, 2

            neighbor_inds = bloat_inds(miss_ray_inds, 1, height, width)
            neighboring_miss_mask = torch.zeros_like(gt_image[..., 0])
            neighboring_miss_mask[neighbor_inds[..., 0], neighbor_inds[...,1]] = 1
            if opt.far_thresh > 0:
                far_ray_mask = (prob_maps["ray_mask"] > 0) * (prob_maps["ray_max_far_dist"] > opt.far_thresh) * (torch.norm(gt_image - prob_maps["coarse_raycolor"], dim=-1, keepdim=True) < 0.1)
                neighboring_miss_mask += far_ray_mask.squeeze(-1)
            neighboring_miss_mask = (prob_maps["ray_mask"].squeeze(-1) > 0) * neighboring_miss_mask * (prob_maps["ray_max_shading_opacity"].squeeze(-1) > opacity_thresh) > 0


            add_xyz = torch.cat([add_xyz, prob_maps["ray_max_sample_loc_w"][neighboring_miss_mask]], dim=0)
            add_conf = torch.cat([add_conf, prob_maps["shading_avg_conf"][neighboring_miss_mask]], dim=0) * opt.prob_mul if prob_maps["shading_avg_conf"] is not None else None
            add_color = torch.cat([add_color, prob_maps["shading_avg_color"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_color"] is not None else None
            add_dir = torch.cat([add_dir, prob_maps["shading_avg_dir"][neighboring_miss_mask]], dim=0) if prob_maps["shading_avg_dir"] is not None else None
            add_embedding = torch.cat([add_embedding, prob_maps["shading_avg_embedding"][neighboring_miss_mask]], dim=0)

            if len(add_xyz) > -1:
                output = prob_maps["coarse_raycolor"].permute(2,0,1)[None, None,...]
                visualizer.save_ref_views({"images": output}, i, subdir="prob_img_{:04d}".format(test_steps))
    model.opt.kernel_size = kernel_size
    if opt.bgmodel.startswith("planepoints"):
        mask = dataset.filter_plane(add_xyz)
        first_lst, _ = masking(mask, [add_xyz, add_embedding, add_color, add_dir, add_conf], [])
        add_xyz, add_embedding, add_color, add_dir, add_conf = first_lst
    if len(add_xyz) > 0:
        visualizer.save_neural_points("prob{:04d}".format(test_steps), add_xyz, None, None, save_ref=False)
        visualizer.print_details("vis added points to probe folder")
    if opt.prob_mode == 0 and opt.prob_num_step > 1:
        model.reset_ray_miss_ranking()
    del visualizer, prob_maps
    model.opt.prob = 0

    return add_xyz, add_embedding, add_color, add_dir, add_conf

def bloat_inds(inds, shift, height, width):
    inds = inds[:,None,:]
    sx, sy = torch.meshgrid(torch.arange(-shift, shift+1, dtype=torch.long), torch.arange(-shift, shift+1, dtype=torch.long))
    shift_inds = torch.stack([sx, sy],dim=-1).reshape(1, -1, 2).cuda()
    inds = inds + shift_inds
    inds = inds.reshape(-1, 2)
    inds[...,0] = torch.clamp(inds[...,0], min=0, max=height-1)
    inds[...,1] = torch.clamp(inds[...,1], min=0, max=width-1)
    return inds

def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def create_all_bg(dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, dummy=False):
    total_num = dataset.total
    height = dataset.height
    width = dataset.width
    bg_ray_lst = []
    random_sample = dataset.opt.random_sample
    for i in range(0, total_num):
        dataset.opt.random_sample = "no_crop"
        if dummy:
            data = dataset.get_dummyrot_item(i)
        else:
            data = dataset.get_item(i)
        raydir = data['raydir'].clone()
        # print("data['pixel_idx']",data['pixel_idx'].shape) # 1, 512, 640, 2
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        start=0
        end = height * width

        data['raydir'] = raydir[:, start:end, :]
        data["pixel_idx"] = pixel_idx[:, start:end, :]
        model.set_input(data)

        xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
        bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"])
        bg_ray = bg_ray.reshape(bg_ray.shape[0], height, width, 3) # 1, 512, 640, 3
        bg_ray_lst.append(bg_ray)
    dataset.opt.random_sample = random_sample
    return bg_ray_lst

def main():
    torch.backends.cudnn.benchmark = True

    opt = TrainOptions().parse()
    cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
                              gpu_ids else torch.device('cpu'))
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
    train_dataset = create_dataset(opt)
    normRw2c = train_dataset.norm_w2c[:3,:3] # torch.eye(3, device="cuda") #
    img_lst=None
    best_PSNR=0.0
    best_iter=0
    points_xyz_all=None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        if len([n for n in glob.glob(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth") if os.path.isfile(n)]) > 0:
            if opt.bgmodel.endswith("plane"):
                _, _, _, _, _, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)

            resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
            if opt.resume_iter == "best":
                opt.resume_iter = "latest"
            resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
            if resume_iter is None:
                epoch_count = 1
                total_steps = 0
                visualizer.print_details("No previous checkpoints, start from scratch!!!!")
            else:
                opt.resume_iter = resume_iter
                states = torch.load(
                    os.path.join(resume_dir, '{}_states.pth'.format(resume_iter)), map_location=cur_device)
                epoch_count = states['epoch_count']
                total_steps = states['total_steps']
                best_PSNR = states['best_PSNR'] if 'best_PSNR' in states else best_PSNR
                best_iter = states['best_iter'] if 'best_iter' in states else best_iter
                best_PSNR = best_PSNR.item() if torch.is_tensor(best_PSNR) else best_PSNR
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                visualizer.print_details('Continue training from {} epoch'.format(opt.resume_iter))
                visualizer.print_details(f"Iter: {total_steps}")
                visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                del states
            opt.mode = 2
            opt.load_points=1
            opt.resume_dir=resume_dir
            opt.resume_iter = resume_iter
            opt.is_train=True
            model = create_model(opt)
        elif opt.load_points < 1:
            points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)
            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train=True
            opt.mode = 2
            model = create_model(opt)
        else:
            load_points = opt.load_points
            opt.is_train = False
            opt.mode = 1
            opt.load_points = 0
            model = create_model(opt)
            model.setup(opt)
            model.eval()
            if load_points in [1,3]:
                points_xyz_all = train_dataset.load_init_points()
            if load_points == 2:
                points_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=100)
            if load_points == 3:
                depth_xyz_all = train_dataset.load_init_depth_points(device="cuda", vox_res=80)
                print("points_xyz_all",points_xyz_all.shape)
                print("depth_xyz_all", depth_xyz_all.shape)
                filter_res = 100
                pc_grid_id, _, pc_space_min, pc_space_max = mvs_utils.construct_vox_points_ind(points_xyz_all, filter_res)
                d_grid_id, depth_inds, _, _ = mvs_utils.construct_vox_points_ind(depth_xyz_all, filter_res, space_min=pc_space_min, space_max=pc_space_max)
                all_grid= torch.cat([pc_grid_id, d_grid_id], dim=0)
                min_id = torch.min(all_grid, dim=-2)[0]
                max_id = torch.max(all_grid, dim=-2)[0] - min_id
                max_id_lst = (max_id+1).cpu().numpy().tolist()
                mask = torch.ones(max_id_lst, device=d_grid_id.device)
                pc_maskgrid_id = (pc_grid_id - min_id[None,...]).to(torch.long)
                mask[pc_maskgrid_id[...,0], pc_maskgrid_id[...,1], pc_maskgrid_id[...,2]] = 0
                depth_maskinds = (d_grid_id[depth_inds,:] - min_id).to(torch.long)
                depth_maskinds = mask[depth_maskinds[...,0], depth_maskinds[...,1], depth_maskinds[...,2]]
                depth_xyz_all = depth_xyz_all[depth_maskinds > 0]
                visualizer.save_neural_points("dep_filtered", depth_xyz_all, None, None, save_ref=False)
                print("vis depth; after pc mask depth_xyz_all",depth_xyz_all.shape)
                points_xyz_all = [points_xyz_all, depth_xyz_all] if opt.vox_res > 0 else torch.cat([points_xyz_all, depth_xyz_all],dim=0)
                del depth_xyz_all, depth_maskinds, mask, pc_maskgrid_id, max_id_lst, max_id, min_id, all_grid

            if opt.ranges[0] > -99.0:
                ranges = torch.as_tensor(opt.ranges, device=points_xyz_all.device, dtype=torch.float32)
                mask = torch.prod(
                    torch.logical_and(points_xyz_all[..., :3] >= ranges[None, :3], points_xyz_all[..., :3] <= ranges[None, 3:]),
                    dim=-1) > 0
                points_xyz_all = points_xyz_all[mask]


            if opt.vox_res > 0:
                points_xyz_all = [points_xyz_all] if not isinstance(points_xyz_all, list) else points_xyz_all
                points_xyz_holder = torch.zeros([0,3], dtype=points_xyz_all[0].dtype, device="cuda")
                for i in range(len(points_xyz_all)):
                    points_xyz = points_xyz_all[i]
                    vox_res = opt.vox_res // (1.5**i)
                    print("load points_xyz", points_xyz.shape)
                    _, sparse_grid_idx, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(points_xyz.cuda() if len(points_xyz) < 80000000 else points_xyz[::(len(points_xyz) // 80000000 + 1), ...].cuda(), vox_res)
                    points_xyz = points_xyz[sampled_pnt_idx, :]
                    print("after voxelize:", points_xyz.shape)
                    points_xyz_holder = torch.cat([points_xyz_holder, points_xyz], dim=0)
                points_xyz_all = points_xyz_holder



            if opt.resample_pnts > 0:
                if opt.resample_pnts == 1:
                    print("points_xyz_all",points_xyz_all.shape)
                    inds = torch.min(torch.norm(points_xyz_all, dim=-1, keepdim=True), dim=0)[1] # use the point closest to the origin
                else:
                    inds = torch.randperm(len(points_xyz_all))[:opt.resample_pnts, ...]
                points_xyz_all = points_xyz_all[inds, ...]

            campos, camdir = train_dataset.get_campos_ray()
            cam_ind = nearest_view(campos, camdir, points_xyz_all, train_dataset.id_list)
            unique_cam_ind = torch.unique(cam_ind)
            print("unique_cam_ind", unique_cam_ind.shape)
            points_xyz_all = [points_xyz_all[cam_ind[:,0]==unique_cam_ind[i], :] for i in range(len(unique_cam_ind))]

            featuredim = opt.point_features_dim
            points_embedding_all = torch.zeros([1, 0, featuredim], device=unique_cam_ind.device, dtype=torch.float32)
            points_color_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_dir_all = torch.zeros([1, 0, 3], device=unique_cam_ind.device, dtype=torch.float32)
            points_conf_all = torch.zeros([1, 0, 1], device=unique_cam_ind.device, dtype=torch.float32)
            print("extract points embeding & colors", )
            for i in tqdm(range(len(unique_cam_ind))):
                id = unique_cam_ind[i]
                batch = train_dataset.get_item(id, full_img=True)
                HDWD = [train_dataset.height, train_dataset.width]
                c2w = batch["c2w"][0].cuda()
                w2c = torch.inverse(c2w)
                intrinsic = batch["intrinsic"].cuda()
                # cam_xyz_all 252, 4
                cam_xyz_all = (torch.cat([points_xyz_all[i], torch.ones_like(points_xyz_all[i][...,-1:])], dim=-1) @ w2c.transpose(0,1))[..., :3]
                embedding, color, dir, conf = model.query_embedding(HDWD, cam_xyz_all[None,...], None, batch['images'].cuda(), c2w[None, None,...], w2c[None, None,...], intrinsic[:, None,...], 0, pointdir_w=True)
                conf = conf * opt.default_conf if opt.default_conf > 0 and opt.default_conf < 1.0 else conf
                points_embedding_all = torch.cat([points_embedding_all, embedding], dim=1)
                points_color_all = torch.cat([points_color_all, color], dim=1)
                points_dir_all = torch.cat([points_dir_all, dir], dim=1)
                points_conf_all = torch.cat([points_conf_all, conf], dim=1)
                # visualizer.save_neural_points(id, cam_xyz_all, color, batch, save_ref=True)
            points_xyz_all=torch.cat(points_xyz_all, dim=0)
            visualizer.save_neural_points("init", points_xyz_all, points_color_all, None, save_ref=load_points == 0)
            print("vis")
            # visualizer.save_neural_points("cam", campos, None, None, None)
            # print("vis")
            # exit()

            opt.resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(opt.resume_dir)
            opt.is_train = True
            opt.mode = 2
            model = create_model(opt)

        if points_xyz_all is not None:
            if opt.bgmodel.startswith("planepoints"):
                gen_pnts, gen_embedding, gen_dir, gen_color, gen_conf = train_dataset.get_plane_param_points()
                visualizer.save_neural_points("pl", gen_pnts, gen_color, None, save_ref=False)
                print("vis pl")
                points_xyz_all = torch.cat([points_xyz_all, gen_pnts], dim=0)
                points_embedding_all = torch.cat([points_embedding_all, gen_embedding], dim=1)
                points_color_all = torch.cat([points_color_all, gen_dir], dim=1)
                points_dir_all = torch.cat([points_dir_all, gen_color], dim=1)
                points_conf_all = torch.cat([points_conf_all, gen_conf], dim=1)
            model.set_points(points_xyz_all.cuda(), points_embedding_all.cuda(), points_color=points_color_all.cuda(),
                             points_dir=points_dir_all.cuda(), points_conf=points_conf_all.cuda(),
                             Rw2c=normRw2c.cuda() if opt.load_points < 1 and opt.normview != 3 else None)
            epoch_count = 1
            total_steps = 0
            del points_xyz_all, points_embedding_all, points_color_all, points_dir_all, points_conf_all

    model.setup(opt, train_len=len(train_dataset))
    model.train()
    data_loader = create_data_loader(opt, dataset=train_dataset)
    dataset_size = len(data_loader)
    visualizer.print_details('# training images = {}'.format(dataset_size))

    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"

    with open('/tmp/.neural-volumetric.name', 'w') as f:
        f.write(opt.name + '\n')

    visualizer.reset()
    if total_steps > 0:
        for scheduler in model.schedulers:
            for i in range(total_steps):
                scheduler.step()
    fg_masks = None
    bg_ray_train_lst, bg_ray_test_lst = [], []
    if opt.bgmodel.endswith("plane"):
        test_dataset = create_dataset(test_opt)
        bg_ray_train_lst = create_all_bg(train_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        bg_ray_test_lst = create_all_bg(test_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        test_bg_info = [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_test_lst]
        del test_dataset
        if opt.vid > 0:
            render_dataset = create_render_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
            bg_ray_render_lst  = create_all_bg(render_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, dummy=True)
            render_bg_info =  [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_render_lst]
    else:
        test_bg_info, render_bg_info = None, None
        img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = None, None, None, None, None

    ############ initial test ###############
    if total_steps == 0 and opt.maximum_step <= 0:
        with torch.no_grad():
            test_opt.nerf_splits = ["test"]
            test_opt.split = "test"
            test_opt.name = opt.name + "/test_{}".format(total_steps)
            test_opt.test_num_step = opt.test_num_step
            test_dataset = create_dataset(test_opt)
            model.opt.is_train = 0
            model.opt.no_loss = 1
            test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps)
            model.opt.no_loss = 0
            model.opt.is_train = 1
            model.train()
            exit()

    if total_steps == 0 and (len(train_dataset.id_list) > 30 or len(train_dataset.view_id_list)  > 30):
        other_states = {
            'epoch_count': 0,
            'total_steps': total_steps,
        }
        model.save_networks(total_steps, other_states)
        visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, 0, total_steps))

    real_start=total_steps
    train_random_sample_size = opt.random_sample_size
    for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        for i, data in enumerate(data_loader):
            if opt.maximum_step is not None and total_steps >= opt.maximum_step:
                break
            if opt.prune_iter > 0 and real_start != total_steps and total_steps % opt.prune_iter == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0 and total_steps <= opt.prune_max_iter:
                with torch.no_grad():
                    model.clean_optimizer()
                    model.clean_scheduler()
                    model.prune_points(opt.prune_thresh)
                    model.setup_optimizer(opt)
                    model.init_scheduler(total_steps, opt)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

            if opt.prob_freq > 0 and real_start != total_steps and total_steps % opt.prob_freq == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0:
                if opt.prob_kernel_size is not None:
                    tier = np.sum(np.asarray(opt.prob_tiers) < total_steps)
                if (model.top_ray_miss_loss[0] > 1e-5 or opt.prob_mode != 0 or opt.far_thresh > 0) and (opt.prob_kernel_size is None or tier < (len(opt.prob_kernel_size) // 3)):
                    torch.cuda.empty_cache()
                    model.opt.is_train = 0
                    model.opt.no_loss = 1
                    with torch.no_grad():
                        prob_opt = copy.deepcopy(test_opt)
                        prob_opt.name = opt.name
                        # if opt.prob_type=0:
                        train_dataset.opt.random_sample = "no_crop"
                        if opt.prob_mode <= 0:
                            train_dataset.opt.random_sample_size = min(32, train_random_sample_size)
                            prob_dataset = train_dataset
                        elif opt.prob_mode == 1:
                            prob_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=1)
                        else:
                            prob_dataset = create_comb_dataset(test_opt, opt, total_steps, test_num_step=1)
                        model.eval()
                        add_xyz, add_embedding, add_color, add_dir, add_conf = probe_hole(model, prob_dataset, Visualizer(prob_opt), prob_opt, None, test_steps=total_steps, opacity_thresh=opt.prob_thresh)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        if opt.prob_mode != 0:
                            del prob_dataset
                        # else:
                        if len(add_xyz) > 0:
                            print("len(add_xyz)", len(add_xyz))
                            model.clean_optimizer_scheduler()
                            model.grow_points(add_xyz, add_embedding, add_color, add_dir, add_conf)
                            length_added = len(add_xyz)
                            del add_xyz, add_embedding, add_color, add_dir, add_conf
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            other_states = {
                                "best_PSNR": best_PSNR,
                                "best_iter": best_iter,
                                'epoch_count': epoch,
                                'total_steps': total_steps,
                            }
                            visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                            print("other_states",other_states)
                            model.save_networks(total_steps, other_states, back_gpu=False)
                            visualizer.print_details(
                                "$$$$$$$$$$$$$$$$$$$$$$$$$$           add grow new points num: {}, all num: {}           $$$$$$$$$$$$$$$$".format(length_added, len(model.neural_points.xyz)))
                            # model.reset_optimizer(opt)
                            # model.reset_scheduler(total_steps, opt)
                            # model.cleanup()
                            # pprint(vars(model))
                            # del model
                            # visualizer.reset()
                            # gc.collect()
                            # torch.cuda.synchronize()
                            # torch.cuda.empty_cache()
                            # input("Press Enter to continue...")
                            # opt.is_train = 1
                            # opt.no_loss = 0
                            # model = create_model(opt)
                            #
                            # model.setup(opt, train_len=len(train_dataset))
                            # model.train()
                            #
                            # if total_steps > 0:
                            #     for scheduler in model.schedulers:
                            #         for i in range(total_steps):
                            #             scheduler.step()

                            exit()

                        visualizer.print_details("$$$$$$$$$$$$$$$$$$$$$$$$$$         add grow new points num: {}, all num: {} $$$$$$$$$$$$$$$$".format(len(add_xyz), len(model.neural_points.xyz)))
                        train_dataset.opt.random_sample = "random"
                    model.train()
                    model.opt.no_loss = 0
                    model.opt.is_train = 1
                    train_dataset.opt.random_sample_size = train_random_sample_size
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                else:
                    visualizer.print_details(
                        'nothing to probe, max ray miss is only {}'.format(model.top_ray_miss_loss[0]))


            total_steps += 1
            model.set_input(data)
            if opt.bgmodel.endswith("plane"):
                if len(bg_ray_train_lst) > 0:
                    bg_ray_all = bg_ray_train_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(model.input)
                    bg_ray, fg_masks = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks=fg_masks)
                data["bg_ray"] = bg_ray
            model.optimize_parameters(total_steps=total_steps)

            losses = model.get_current_losses()
            visualizer.accumulate_losses(losses)

            if opt.lr_policy.startswith("iter"):
                model.update_learning_rate(opt=opt, total_steps=total_steps)

            if total_steps and total_steps % opt.print_freq == 0:
                if opt.show_tensorboard:
                    visualizer.plot_current_losses_with_tb(total_steps, losses)
                visualizer.print_losses(total_steps)
                visualizer.reset()

            if hasattr(opt, "save_point_freq") and total_steps and total_steps % opt.save_point_freq == 0 and (opt.prune_iter > 0 and total_steps <= opt.prune_max_iter or opt.save_point_freq==1):
                visualizer.save_neural_points(total_steps, model.neural_points.xyz, model.neural_points.points_embeding, data, save_ref=opt.load_points==0)
                visualizer.print_details('saving neural points at total_steps {})'.format(total_steps))

            try:
                if total_steps == 10000 or (total_steps % opt.save_iter_freq == 0 and total_steps > 0):
                    other_states = {
                        "best_PSNR": best_PSNR,
                        "best_iter": best_iter,
                        'epoch_count': epoch,
                        'total_steps': total_steps,
                    }
                    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
                    model.save_networks(total_steps, other_states)
            except Exception as e:
                visualizer.print_details(e)


            if opt.vid > 0 and total_steps % opt.vid == 0 and total_steps > 0:
                torch.cuda.empty_cache()
                test_dataset = create_render_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
                model.opt.is_train = 0
                model.opt.no_loss = 1
                with torch.no_grad():
                    render_vid(model, test_dataset, Visualizer(test_opt), test_opt, render_bg_info, steps=total_steps)
                model.opt.no_loss = 0
                model.opt.is_train = 1
                del test_dataset

            if total_steps == 10000 or (total_steps % opt.test_freq == 0 and total_steps < (opt.maximum_step - 1) and total_steps > 0):
                torch.cuda.empty_cache()
                test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=opt.test_num_step)
                model.opt.is_train = 0
                model.opt.no_loss = 1
                with torch.no_grad():
                    if opt.test_train == 0:
                        test_psnr = test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=True)
                    else:
                        train_dataset.opt.random_sample = "no_crop"
                        test_psnr = test(model, train_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, lpips=True)
                        train_dataset.opt.random_sample = opt.random_sample
                model.opt.no_loss = 0
                model.opt.is_train = 1
                del test_dataset
                best_iter = total_steps if test_psnr > best_PSNR else best_iter
                best_PSNR = max(test_psnr, best_PSNR)
                visualizer.print_details(f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
            model.train()

        # try:
        #     print("saving the model at the end of epoch")
        #     other_states = {'epoch_count': epoch, 'total_steps': total_steps}
        #     model.save_networks('latest', other_states)
        #
        # except Exception as e:
        #     print(e)

        if opt.maximum_step is not None and total_steps >= opt.maximum_step:
            visualizer.print_details('{}: End of stepts {} / {} \t Time Taken: {} sec'.format(
                opt.name, total_steps, opt.maximum_step,
                time.time() - epoch_start_time))
            break

    del train_dataset
    other_states = {
        'epoch_count': epoch,
        'total_steps': total_steps,
    }
    visualizer.print_details('saving model ({}, epoch {}, total_steps {})'.format(opt.name, epoch, total_steps))
    model.save_networks(total_steps, other_states)

    torch.cuda.empty_cache()
    test_dataset = create_test_dataset(test_opt, opt, total_steps, test_num_step=1)
    model.opt.no_loss = 1
    model.opt.is_train = 0

    visualizer.print_details("full datasets test:")
    with torch.no_grad():
        test_psnr = test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=total_steps, gen_vid=True, lpips=True)
    best_iter = total_steps if test_psnr > best_PSNR else best_iter
    best_PSNR = max(test_psnr, best_PSNR)
    visualizer.print_details(
        f"test at iter {total_steps}, PSNR: {test_psnr}, best_PSNR: {best_PSNR}, best_iter: {best_iter}")
    exit()


def save_points_conf(visualizer, xyz, points_color, points_conf, total_steps):
    print("total:", xyz.shape, points_color.shape, points_conf.shape)
    colors, confs = points_color[0], points_conf[0,...,0]
    pre = -1000
    for i in range(12):
        thresh = (i * 0.1) if i <= 10 else 1000
        mask = ((confs <= thresh) * (confs > pre)) > 0
        thresh_xyz = xyz[mask, :]
        thresh_color = colors[mask, :]
        visualizer.save_neural_points(f"{total_steps}-{thresh}", thresh_xyz, thresh_color[None, ...], None, save_ref=False)
        pre = thresh
    exit()


def create_render_dataset(test_opt, opt, total_steps, test_num_step=1):
    test_opt.nerf_splits = ["render"]
    test_opt.split = "render"
    test_opt.name = opt.name + "/vid_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_opt.random_sample_size = 30
    test_dataset = create_dataset(test_opt)
    return test_dataset


def create_test_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.name = opt.name + "/test_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset


def create_comb_dataset(test_opt, opt, total_steps, prob=None, test_num_step=1):
    test_opt.prob = prob if prob is not None else test_opt.prob
    test_opt.nerf_splits = ["comb"]
    test_opt.split = "comb"
    test_opt.name = opt.name + "/comb_{}".format(total_steps)
    test_opt.test_num_step = test_num_step
    test_dataset = create_dataset(test_opt)
    return test_dataset

if __name__ == '__main__':
    main()
