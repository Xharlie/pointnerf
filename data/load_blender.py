import os
import numpy as np
import imageio 
import json
import torch
import pickle, random

# trans_t = lambda t : tf.convert_to_tensor([
#     [1,0,0,0],
#     [0,1,0,0],
#     [0,0,1,t],
#     [0,0,0,1],
# ], dtype=tf.float32)
#
# rot_phi = lambda phi : tf.convert_to_tensor([
#     [1,0,0,0],
#     [0,tf.cos(phi),-tf.sin(phi),0],
#     [0,tf.sin(phi), tf.cos(phi),0],
#     [0,0,0,1],
# ], dtype=tf.float32)
#
# rot_theta = lambda th : tf.convert_to_tensor([
#     [tf.cos(th),0,-tf.sin(th),0],
#     [0,1,0,0],
#     [tf.sin(th),0, tf.cos(th),0],
#     [0,0,0,1],
# ], dtype=tf.float32)

trans_t = lambda t : np.asarray([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=np.float32)

rot_phi = lambda phi : np.asarray([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=np.float32)

rot_theta = lambda th : np.asarray([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=np.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w


blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def load_blender_data(basedir, splits, half_res=False, testskip=1):
    splits = ['train', 'val', 'test'] if splits is None else splits
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']) @ blender2opencv)

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(len(splits))]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    stride = 20
    render_poses = np.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, stride+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.

    intrinsic = np.asarray([[focal, 0, W/2],
                 [0, focal, H/2],
                 [0,0,1]])
    return imgs, poses, render_poses, [H, W, focal], i_split, intrinsic


def load_blender_cloud(point_path, point_num):
    point_norms = None
    with open(point_path, 'rb') as f:
        print("point_file_path ################", point_path)
        all_infos = pickle.load(f)
        point_xyz = all_infos["point_xyz"]
        if "point_face_normal" in all_infos:
            point_norms = all_infos["point_face_normal"]
    print("surface point cloud ",len(point_xyz), "mean pos:", np.mean(point_xyz, axis=0), "min pos:",np.min(point_xyz, axis=0), "mean max:",np.max(point_xyz, axis=0))
    if point_num < len(point_xyz):
        inds = np.asarray(random.choices(range(len(point_xyz)), k=point_num))
        point_norms = point_norms[inds, :] if point_norms is not None else None
        return point_xyz[inds, :], point_norms
    else:
        return point_xyz, point_norms
