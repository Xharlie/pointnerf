import sys
import os
import pathlib
# sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import torch.nn.functional as F

import copy
import torch
import numpy as np
import time
from models.mvs import mvs_utils
from tqdm import tqdm

import cv2
from PIL import Image

def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)

    oor_mask = np.logical_or(np.logical_or(x_src >= width, x_src < 0),np.logical_or(y_src >= height, y_src < 0))

    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # print("depth_src",depth_src.shape, x_src.shape, y_src.shape)
    # sampled_depth_src=depth_src
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, oor_mask



def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, oor_mask = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    # H, W
    mask = np.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, ~oor_mask, depth_reprojected, x2d_src, y2d_src



def filter_by_masks(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt):
    xyz_world_lst=[]
    xyz_ref_lst=[]
    confidence_filtered_lst = []
    B, N, H, W, _ = cam_xyz_all[0].shape
    cam_xyz_all = [cam_xyz.reshape(H, W, 3) for cam_xyz in cam_xyz_all]
    count = 0
    for ref_view in tqdm(range(len(cam_xyz_all))):
        ref_intrinsics, ref_extrinsics, ref_cam_xy, ref_depth_est, confidence, points_mask = intrinsics_all[ref_view][0], extrinsics_all[ref_view][0], cam_xyz_all[ref_view][...,:-1], cam_xyz_all[ref_view][...,-1], confidence_all[ref_view][0,0,...], points_mask_all[ref_view][0,0,...]
        photo_mask = confidence > opt.depth_conf_thresh
        sum_srcview_depth_ests = 0
        geo_mask_sum = 0
        visible_and_match_sum = 0
        visible_sum = 0

        # compute the geometric mask
        for src_view in range(len(cam_xyz_all)):
            if ref_view == src_view:
                continue
            src_intrinsics, src_extrinsics, src_depth_est = intrinsics_all[src_view][0], extrinsics_all[src_view][0], cam_xyz_all[src_view][...,-1]
            geo_mask, vis_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics, ref_extrinsics, src_depth_est, src_intrinsics, src_extrinsics)
            visible_sum += vis_mask.astype(np.float32)
            visible_and_match_sum += np.logical_and(vis_mask, geo_mask).astype(np.float32)
            geo_mask_sum += geo_mask.astype(np.int32)
            sum_srcview_depth_ests += depth_reprojected
        depth_est_averaged = (sum_srcview_depth_ests + ref_depth_est) / (geo_mask_sum + 1)
        # at least 3 source views matched
        geo_mask = geo_mask_sum >= opt.geo_cnsst_num

        final_mask = np.logical_and(np.logical_and(photo_mask, geo_mask), points_mask)

        # vis_geo_mask = np.divide(visible_and_match_sum, visible_sum, out=np.ones_like(visible_and_match_sum), where=visible_sum!=0) > 0.05
        # final_mask = np.logical_and(np.logical_and(photo_mask, vis_geo_mask), points_mask)

        xy, depth = ref_cam_xy[final_mask,:], depth_est_averaged[final_mask][...,None]
        xyz_ref = np.concatenate([xy, depth], axis=-1)
        xyz_world = np.matmul(np.concatenate([xyz_ref, np.ones_like(xyz_ref[...,0:1])], axis=-1), np.transpose(np.linalg.inv(ref_extrinsics)))[:,:3]
        confidence_filtered = confidence[final_mask]
        xyz_world, xyz_ref, confidence_filtered = range_mask_np(xyz_world, xyz_ref, confidence_filtered, opt)

        xyz_world_lst.append(xyz_world)
        xyz_ref_lst.append(xyz_ref)
        confidence_filtered_lst.append(confidence_filtered)

    return xyz_ref_lst, xyz_world_lst, confidence_filtered_lst

def range_mask_lst_np(xyz_world_all, cam_xyz_all, confidence_filtered_lst, opt):
    if opt.ranges[0] > -99.0:
        for i in range(len(xyz_world_all)):
            xyz_world, cam_xyz, confidence_filtered = range_mask_np(xyz_world_all[i], cam_xyz_all[i], confidence_filtered_lst[i], opt)
            xyz_world_all[i], cam_xyz_all[i], confidence_filtered_lst[i] = xyz_world, cam_xyz, confidence_filtered
    return xyz_world_all, cam_xyz_all, confidence_filtered_lst


def range_mask_np(xyz_world, xyz_ref, confidence_filtered, opt):
    # print("range_mask_np")
    if opt.ranges[0] > -99.0:
        ranges = np.asarray(opt.ranges)
        mask = np.prod(np.logical_and(xyz_world >= ranges[None, :3], xyz_world <= ranges[None, 3:]), axis=-1) > 0
        xyz_world = xyz_world[mask]
        xyz_ref = xyz_ref[mask]
        confidence_filtered = confidence_filtered[mask]
    return xyz_world, xyz_ref, confidence_filtered


def range_mask_torch(xyz_world, xyz_ref, confidence_filtered, opt):
    # print("range_mask_torch")
    if opt.ranges[0] > -99.0:
        ranges = torch.as_tensor(opt.ranges, device=xyz_world.device, dtype=torch.float32)
        mask = torch.prod(torch.logical_and(xyz_world[..., :3] >= ranges[None, :3], xyz_world[..., :3] <= ranges[None, 3:]), dim=-1) > 0
        xyz_world = xyz_world[mask]
        xyz_ref = xyz_ref[mask]
        confidence_filtered = confidence_filtered[mask]
    return xyz_world, xyz_ref, confidence_filtered


def reproject_with_depth_gpu(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height, device=depth_ref.device), torch.arange(0, width, device=depth_ref.device))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space

    xyz_ref = torch.matmul(torch.linalg.inv(intrinsics_ref),
                        torch.stack([x_ref, y_ref, torch.ones_like(x_ref, device=x_ref.device)], dim=0) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.linalg.inv(extrinsics_ref)),
                           torch.cat([xyz_ref, torch.ones_like(x_ref)[None,:]], dim=0))[:3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).to(torch.float32)
    y_src = xy_src[1].reshape([height, width]).to(torch.float32)

    oor_mask = torch.logical_or(torch.logical_or(x_src >= width, x_src < 0), torch.logical_or(y_src >= height, y_src < 0))

    # sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    sampled_depth_src =  F.grid_sample(depth_src[None, None, ...], torch.stack([x_src * 2 / (width-1) - 1, y_src * 2 / (height-1) - 1], dim=-1)[None,...], align_corners=True, mode='bilinear', padding_mode='border')

    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.linalg.inv(intrinsics_src), torch.cat([xy_src, torch.ones_like(x_ref)[None,:]], dim=0) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.linalg.inv(extrinsics_src)),
                               torch.cat([xyz_src, torch.ones_like(x_ref)[None,:]], dim=0))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).to(torch.float32)
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).to(torch.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).to(torch.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src, oor_mask



def check_geometric_consistency_gpu(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height, device=depth_ref.device),
                                  torch.arange(0, width, device=depth_ref.device))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src, oor_mask = reproject_with_depth_gpu(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref
    # H, W
    mask = torch.logical_and(dist < 1, relative_depth_diff < 0.01)
    depth_reprojected[~mask] = 0

    return mask, ~oor_mask, depth_reprojected, x2d_src, y2d_src



def filter_by_masks_gpu(cam_xyz_all, intrinsics_all, extrinsics_all, confidence_all, points_mask_all, opt, vis=False, return_w=False, cpu2gpu=False, near_fars_all=None):
    xyz_cam_lst=[]
    xyz_world_lst=[]
    confidence_filtered_lst = []
    B, N, C, H, W, _ = cam_xyz_all[0].shape
    cam_xyz_all = [cam_xyz.view(C,H,W,3) for cam_xyz in cam_xyz_all]
    for cam_view in tqdm(range(len(cam_xyz_all))) if vis else range(len(cam_xyz_all)):
        near_fars = near_fars_all[cam_view] if near_fars_all is not None else None
        if opt.manual_depth_view > 1:
            xyz_cam, cam_xy, confidence, points_mask, cam_extrinsics = cam_xyz_all[cam_view], cam_xyz_all[cam_view][0, :, :, :-1], confidence_all[cam_view][0,...], points_mask_all[cam_view][0,...], extrinsics_all[cam_view][0]
            final_mask = torch.logical_and(confidence > opt.depth_conf_thresh, points_mask)
            xyz_cam = xyz_cam[final_mask]
            confidence *= 0.3
        else:
            cam_intrinsics, cam_extrinsics, cam_xy, cam_depth_est, confidence, points_mask = intrinsics_all[cam_view][0], extrinsics_all[cam_view][0], cam_xyz_all[cam_view][0,...,:-1], cam_xyz_all[cam_view][0,...,-1], confidence_all[cam_view][0,0,...], points_mask_all[cam_view][0,0,...]
            if cpu2gpu:
                cam_xy, cam_depth_est, confidence, points_mask = cam_xy.cuda(), cam_depth_est.cuda(), confidence.cuda(), points_mask.cuda()
            sum_srcview_depth_ests = 0
            geo_mask_sum = 0
            visible_and_match_sum = 0
            visible_and_not_match_sum = 0
            visible_sum = 0

            # compute the geometric mask
            for src_view in range(len(cam_xyz_all)):
                if cam_view == src_view:
                    continue
                src_intrinsics, src_extrinsics, src_depth_est = intrinsics_all[src_view][0], extrinsics_all[src_view][0], cam_xyz_all[src_view][0,...,-1]
                if cpu2gpu:
                    src_depth_est = src_depth_est.cuda()
                geo_mask, vis_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency_gpu(cam_depth_est, cam_intrinsics, cam_extrinsics, src_depth_est, src_intrinsics, src_extrinsics)
                visible_sum += vis_mask.to(torch.float32)
                visible_and_match_sum += torch.logical_and(vis_mask, geo_mask).to(torch.float32)
                visible_and_not_match_sum += torch.logical_and(vis_mask, ~geo_mask).to(torch.float32)
                geo_mask_sum += geo_mask.to(torch.int32)
                sum_srcview_depth_ests += depth_reprojected

            depth_est_averaged = (sum_srcview_depth_ests + cam_depth_est) / (geo_mask_sum + 1)
            # at least 3 source views matched
            geo_mask = geo_mask_sum >= opt.geo_cnsst_num # visible_and_not_match_sum < 3 #
            final_mask = torch.logical_and(confidence > opt.depth_conf_thresh, points_mask)
            final_mask = torch.logical_and(final_mask, geo_mask) if len(cam_xyz_all)>1 else final_mask
            xy, depth = cam_xy[final_mask,:], depth_est_averaged[final_mask][...,None]
            xyz_cam = torch.cat([xy, depth], dim=-1)

        confidence_filtered = confidence[final_mask]
        if opt.default_conf > 1.0:
            assert opt.manual_depth_view <= 1
            confidence_filtered = reassign_conf(confidence_filtered, final_mask, geo_mask_sum, opt.geo_cnsst_num)

        if opt.far_plane_shift is not None:
            assert near_fars is not None
            bg_mask = ~final_mask if final_mask.dim() == 2 else (torch.sum(final_mask, dim=0) < 1)
            bg_xy = cam_xy[bg_mask,:]

            xyz_cam_extra = torch.cat([bg_xy, torch.ones_like(bg_xy[...,:1]) * near_fars[1] + opt.far_plane_shift], dim=-1)
            xyz_cam = torch.cat([xyz_cam, xyz_cam_extra], dim=0)
            confidence_extra = torch.ones_like(xyz_cam_extra[...,-1]) * 0.02
            confidence_filtered = torch.cat([confidence_filtered, confidence_extra], dim=0)

        xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(cam_extrinsics).transpose(0,1)
        # print("xyz_world",xyz_world.shape)

        xyz_world, xyz_cam, confidence_filtered = range_mask_torch(xyz_world, xyz_cam, confidence_filtered, opt)

        xyz_cam_lst.append(xyz_cam.cpu() if cpu2gpu else xyz_cam)
        xyz_world_lst.append(xyz_world[:,:3].cpu() if cpu2gpu else xyz_world[:,:3])
        confidence_filtered_lst.append(confidence_filtered.cpu() if cpu2gpu else confidence_filtered)

    return xyz_cam_lst, xyz_world_lst, confidence_filtered_lst


def reassign_conf(confidence_filtered, final_mask, geo_mask_sum, geo_cnsst_num):
    geo_mask_sum = geo_mask_sum[final_mask] - geo_cnsst_num + 1
    confidence_filtered *= (1 - 1.0 / torch.pow(1.14869, torch.clamp(geo_mask_sum, min=1, max=10))) # 1.14869 = root 2 by 5
    return confidence_filtered


