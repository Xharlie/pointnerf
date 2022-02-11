from models.mvs.mvs_utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
from kornia import create_meshgrid
import time
import json
from . import data_utils
import glob
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import h5py

from data.base_dataset import BaseDataset
import configparser
import itertools
from os.path import join
import cv2
# import torch.nn.functional as F
from .data_utils import get_dtu_raydir
import copy
FLIP_Z = np.asarray([
    [1,0,0],
    [0,1,0],
    [0,0,-1],
], dtype=np.float32)


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def colorjitter(img, factor):
    # brightness_factor,contrast_factor,saturation_factor,hue_factor
    # img = F.adjust_brightness(img, factor[0])
    # img = F.adjust_contrast(img, factor[1])
    img = F.adjust_saturation(img, factor[2])
    img = F.adjust_hue(img, factor[3]-1.0)

    return img


def unique_lst(list1):
    x = np.array(list1)
    return np.unique(x)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate
    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    c2w = torch.FloatTensor(c2w)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)],
                             -1)  # (H, W, 3)

    return directions


def flip_z(poses):
    z_flip_matrix = np.eye(4, dtype=np.float32)
    z_flip_matrix[2, 2] = -1.0
    return np.matmul(poses, z_flip_matrix[None,...])

class LlffFtDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # ['random', 'random2', 'patch'], default: no random sample
        parser.add_argument('--random_sample',
                            type=str,
                            default='none',
                            help='random sample pixels')
        parser.add_argument('--random_sample_size',
                            type=int,
                            default=1024,
                            help='number of random samples')
        parser.add_argument('--init_view_num',
                            type=int,
                            default=3,
                            help='number of random samples')
        parser.add_argument('--shape_id', type=int, default=0, help='shape id')
        parser.add_argument('--trgt_id', type=int, default=0, help='shape id')
        parser.add_argument('--num_nn',
                            type=int,
                            default=1,
                            help='number of nearest views in a batch')
        parser.add_argument(
            '--near_plane',
            type=float,
            default=2.125,
            help=
            'Near clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument(
            '--far_plane',
            type=float,
            default=4.525,
            help=
            'Far clipping plane, by default it is computed according to the distance of the camera '
        )

        parser.add_argument(
            '--bg_color',
            type=str,
            default="white",
            help=
            'background color, white|black(None)|random|rgb (float, float, float)'
        )

        parser.add_argument(
            '--scan',
            type=str,
            default="scan1",
            help=''
        )
        parser.add_argument(
            '--full_comb',
            type=int,
            default=0,
            help=''
        )

        parser.add_argument('--inverse_gamma_image',
                            type=int,
                            default=-1,
                            help='de-gamma correct the input image')
        parser.add_argument('--pin_data_in_memory',
                            type=int,
                            default=-1,
                            help='load whole data in memory')
        parser.add_argument('--normview',
                            type=int,
                            default=0,
                            help='load whole data in memory')
        parser.add_argument(
            '--id_range',
            type=int,
            nargs=3,
            default=(0, 385, 1),
            help=
            'the range of data ids selected in the original dataset. The default is range(0, 385). If the ids cannot be generated by range, use --id_list to specify any ids.'
        )
        parser.add_argument(
            '--img_wh',
            type=int,
            nargs=2,
            default=(960, 640),
            help='resize target of the image'
        )
        parser.add_argument(
            '--id_list',
            type=int,
            nargs='+',
            default=None,
            help=
            'the list of data ids selected in the original dataset. The default is range(0, 385).'
        )
        parser.add_argument(
            '--split',
            type=str,
            default="train",
            help=
            'train, val, test'
        )
        parser.add_argument("--half_res", action='store_true',
                            help='load blender synthetic data at 400x400 instead of 800x800')
        parser.add_argument("--testskip", type=int, default=8,
                            help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
        parser.add_argument('--dir_norm',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        parser.add_argument('--train_load_num',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')
        parser.add_argument('--holdoff',
                            type=int,
                            default=8,
                            help='normalize the ray_dir to unit length or not, default not')

        return parser



    def initialize(self, opt, downSample=1.0, max_len=-1, norm_w2c=None, norm_c2w=None):
        self.opt = opt
        self.data_dir = opt.data_root
        self.scan = opt.scan
        self.split = opt.split

        self.img_wh = (int(opt.img_wh[0] * downSample), int(opt.img_wh[1] * downSample))
        self.downSample = downSample

        self.scale_factor = 1.0 / 1.0
        self.max_len = max_len

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        if not self.opt.bg_color or self.opt.bg_color == 'black':
            self.bg_color = (0, 0, 0)
        elif self.opt.bg_color == 'white':
            self.bg_color = (1, 1, 1)
        elif self.opt.bg_color == 'random':
            self.bg_color = 'random'
        else:
            self.bg_color = [float(one) for one in self.opt.bg_color.split(",")]

        self.define_transforms()
        self.ori_poses_bounds = np.load(os.path.join(self.data_dir, self.scan, 'poses_bounds.npy'))
        poses, avg_poses, bounds = self.get_poses(self.ori_poses_bounds)
        self.norm_w2c, self.norm_c2w = torch.eye(4, device="cuda", dtype=torch.float32), torch.eye(4, device="cuda", dtype=torch.float32)
        norm_c2w = None
        if opt.normview == 1:
            norm_c2w = avg_poses
        if opt.normview == 2:
            self.norm_w2c, self.norm_c2w = torch.as_tensor(np.linalg.inv(avg_poses), device="cuda", dtype=torch.float32), torch.as_tensor(avg_poses, device="cuda", dtype=torch.float32)
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats(poses, bounds, norm_c2w=norm_c2w)
        self.build_init_metas(opt.holdoff)
        self.load_images()
        self.total = len(self.id_list)
        print("dataset total:", self.split, self.total)
    #
    # def read_images(self):
    #     image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images_4/*')))

    def load_images(self):
        imgs = []
        image_paths = sorted(glob.glob(os.path.join(self.data_dir, self.scan, 'images_4/*')))
        print("id_list", self.id_list, image_paths)
        for i in self.all_id_list:
            img = Image.open(image_paths[i]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            imgs.append(img)
        self.imgs = imgs



    def get_poses(self, poses_bounds):
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        bounds = poses_bounds[:, -2:]  # (N_images, 2)
        # Step 1: rescale focal length according to training resolution
        H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images

        self.focal = [focal * self.img_wh[0] / W, focal * self.img_wh[1] / H]

        # Step 2: correct poses
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        poses, avg_poses = self.center_poses(poses, self.blender2opencv)

        near_original = bounds.min()
        far_original = bounds.max()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        bounds /= scale_factor
        poses[..., 3] /= scale_factor
        avg_poses[..., 3] /= scale_factor
        avg_poses_holder = np.eye(4)
        avg_poses_holder[:3] = avg_poses

        # 2.65 / 200 * 192 = 2.544,  min 2.1250
        # range_original = far_original - near_original
        # scale_factor = range_original / 2.544
        # bounds /= scale_factor
        # poses[..., 3] /= scale_factor
        # avg_poses[..., 3] /= scale_factor
        # avg_poses_holder = np.eye(4)
        # avg_poses_holder[:3] = avg_poses

        return poses, avg_poses_holder, bounds


    def build_proj_mats(self, poses, bounds, norm_c2w=None):
        w, h = self.img_wh

        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        self.all_id_list = range(len(poses))
        self.near_far = [bounds.min() * 0.8, bounds.max() * 1.2]
        # self.near_far = np.asarray([bounds.min()*0.9, bounds.max()*1.1]).astype(np.float32)
        print("dataset near_far", self.near_far)
        for vid in self.all_id_list:
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3] = poses[vid]
            w2c = np.linalg.inv(c2w)
            if norm_c2w is not None:
                w2c = w2c @ norm_c2w
                c2w = np.linalg.inv(w2c)
            cam2worlds.append(c2w)
            world2cams.append(w2c)
            # build proj mat from source views to ref view
            proj_mat_l = np.eye(4)
            intrinsic = np.asarray([[self.focal[0], 0, w / 2], [0, self.focal[1], h / 2], [0, 0, 1]])
            intrinsics.append(intrinsic.copy())
            intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
            proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]
            proj_mats += [[proj_mat_l, bounds[vid]]]
            # proj_mats += [[proj_mat_l, self.near_far]]

        return proj_mats, np.stack(intrinsics), np.stack(world2cams), np.stack(cam2worlds)


    def build_init_metas(self, holdoff):
        self.id_list_test = np.arange(len(self.all_id_list))[::holdoff]
        self.id_list_train = np.array([i for i in np.arange(len(self.all_id_list)) if (i not in self.id_list_test)])
        self.id_list = self.id_list_test if self.split == "test" else self.id_list_train
        self.view_id_list = [] # index is id_list's position e.g., the real image id is  id_list[view_id]
        cam_xyz_lst = [c2w[:3,3] for c2w in self.cam2worlds[self.id_list_train, :, :]]
        test_cam_xyz_lst = [c2w[:3,3] for c2w in self.cam2worlds[self.id_list_test, :, :]]

        if self.split=="train":
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            test_cam_xyz = np.stack(test_cam_xyz_lst, axis=0)
            # if self.opt.full_comb <= 1:
            triangles = data_utils.triangluation_bpa(cam_xyz, test_pnts=test_cam_xyz, full_comb=self.opt.full_comb >=1)
            print("triangles:", triangles.shape)
            if self.opt.full_comb <= 1:
                self.view_id_list = [triangles[i] for i in range(len(triangles))]
            elif self.opt.full_comb == 2: # all combination
                triangles = list(itertools.combinations(range(len(cam_xyz)), 3))
                self.view_id_list = [triangles[i] for i in range(len(triangles))]
            elif self.opt.full_comb in [3,4]: # 1 jump
                triplets = []
                first_dict = {}
                for tris in triangles:
                    if tris[0] not in first_dict.keys():
                        first_dict[tris[0]] = []
                    first_dict[tris[0]] += [tris[1], tris[2]]

                for key, val in first_dict.items():
                    first_dict[key] = list(unique_lst(val))

                if self.opt.full_comb == 3:
                    for key, val in first_dict.items():
                        pairs = list(itertools.combinations(first_dict[key], 2))
                        triplets += [[key]+list(pair) for pair in pairs]
                    self.view_id_list = [triplets[i] for i in range(len(triplets))]
                elif self.opt.full_comb == 4:
                    second_dict = copy.deepcopy(first_dict)
                    for key, val in first_dict.items():
                        for second in val:
                            second_dict[key] += first_dict[second]
                        second_dict[key] = list(unique_lst(second_dict[key]))
                        second_dict[key] = [val for val in second_dict[key] if val != key and val not in first_dict[key]]
                        # print("key val", key, second_dict[key])
                    for key, val in second_dict.items():
                        pairs = list(itertools.combinations(second_dict[key], 2))
                        print("key val", key, pairs)
                        triplets += [[key] + list(pair) for pair in pairs]
                    print("len()", len(triplets))
                    # exit()
                    self.view_id_list = [triplets[i] for i in range(len(triplets))]

                # print("&&&&&&&&&&&&&&&&&&&&&&&&&&self.view_id_list", len(self.view_id_list))
            # elif self.opt.full_comb == 4:  # 1 jump

            # if self.opt.full_comb<0:
            #     with open(f'../data/nerf_synth_configs/list/lego360_init_pairs.txt') as f:
            #         for line in f:
            #             str_lst = line.rstrip().split(',')
            #             src_views = [int(x) for x in str_lst]
            #             self.view_id_list.append(src_views)


    def center_poses(self, poses, blender2opencv):
        """
        Center the poses so that we can use NDC.
        See https://github.com/bmild/nerf/issues/34
        Inputs:
            poses: (N_images, 3, 4)
        Outputs:
            poses_centered: (N_images, 3, 4) the centered poses
            pose_avg: (3, 4) the average pose
        """

        pose_avg = average_poses(poses)  # (3, 4)
        pose_avg_homo = np.eye(4)
        pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
        # by simply adding 0, 0, 0, 1 as the last row
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
        poses_homo = \
            np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

        poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
        poses_centered = poses_centered @ blender2opencv
        poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

        return poses_centered, (np.linalg.inv(pose_avg_homo) @ blender2opencv)[:3, :]



    def define_transforms(self):
        self.transform = T.ToTensor()


    def __len__(self):
        if self.split == 'train':
            return len(self.id_list) if self.max_len <= 0 else self.max_len
        return len(self.id_list) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'NerfSynthFtDataset'


    def __del__(self):
        print("end loading")

    def normalize_rgb(self, data):
        # to unnormalize image for visualization
        # data C, H, W
        C, H, W = data.shape
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (data - mean) / std

    def get_init_item(self, idx, crop=False):
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h = [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = self.id_list[i]
            # mvs_images += [self.normalize_rgb(self.blackimgs[vid])]
            # mvs_images += [self.whiteimgs[vid]]
            # mvs_images += [self.blackimgs[vid]]
            imgs += [self.imgs[vid]]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            near_fars.append(near_far)

        for i in range(len(affine_mat)):
            view_proj_mats = []
            ref_proj_inv = affine_mat_inv[i]
            for j in range(len(affine_mat)):
                if i == j:  # reference view
                    view_proj_mats += [np.eye(4)]
                else:
                    view_proj_mats += [affine_mat[j] @ ref_proj_inv]
            # view_proj_mats: 4, 4, 4
            view_proj_mats = np.stack(view_proj_mats)
            proj_mats.append(view_proj_mats[:, :3])
        # (4, 4, 3, 4)
        proj_mats = np.stack(proj_mats)
        imgs = np.stack(imgs).astype(np.float32)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        # view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        # c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['near_fars_depth'] = self.near_far
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        # sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        # sample['scan'] = scan
        # sample['c2ws_all'] = c2ws_all.astype(np.float32)


        for key, value in sample.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                    sample[key] = value.unsqueeze(0)

        return sample



    def __getitem__(self, id, crop=False):
        item = {}
        vid = self.id_list[id]
        img = self.imgs[vid]
        w2c = self.world2cams[vid]
        c2w = self.cam2worlds[vid]
        intrinsic = self.intrinsics[vid]
        proj_mat_ls, near_far = self.proj_mats[vid]

        gt_image = np.transpose(img, (1,2,0))
        # print("gt_image", gt_image.shape)
        width, height = gt_image.shape[1], gt_image.shape[0]
        camrot = (c2w[0:3, 0:3])
        campos = c2w[0:3, 3]
        # print("camrot", camrot, campos)

        item["intrinsic"] = intrinsic
        # item["intrinsic"] = sample['intrinsics'][0, ...]
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([near_far[1] * 1.1]).view(1, 1)
        item['near'] = torch.FloatTensor([near_far[0] * 0.9]).view(1, 1)
        item['h'] = height
        item['w'] = width
        # item['depths_h'] = self.depths[id]
        # print("near_far", near_far)
        # bounding box

        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(0,
                                   width,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(0,
                                   height,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(0,
                                   width - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(0,
                                   height - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        item["pixel_idx"] = pixelcoords
        # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
        raydir = get_dtu_raydir(pixelcoords, item["intrinsic"], camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        gt_image = gt_image[py.astype(np.int32), px.astype(np.int32)]
        # gt_mask = gt_mask[py.astype(np.int32), px.astype(np.int32), :]
        gt_image = np.reshape(gt_image, (-1, 3))
        item['gt_image'] = gt_image
        item['id'] = vid

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)

        return item



    def get_item(self, idx, crop=False):
        item = self.__getitem__(idx, crop=crop)

        for key, value in item.items():
            if not isinstance(value, str):
                if not torch.is_tensor(value):
                    value = torch.as_tensor(value)
                item[key] = value.unsqueeze(0)
        return item



    def get_dummyrot_item(self, idx, crop=False):

        item = {}
        width, height = self.width, self.height

        transform_matrix = self.render_poses[idx]
        camrot = (transform_matrix[0:3, 0:3])
        campos = transform_matrix[0:3, 3]
        focal = self.focal

        item["focal"] = focal
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float()
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        # near far
        if self.opt.near_plane is not None:
            near = self.opt.near_plane
        else:
            near = max(dist - 1.5, 0.02)
        if self.opt.far_plane is not None:
            far = self.opt.far_plane  # near +
        else:
            far = dist + 0.7
        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([far]).view(1, 1)
        item['near'] = torch.FloatTensor([near]).view(1, 1)
        item['h'] = self.height
        item['w'] = self.width


        subsamplesize = self.opt.random_sample_size
        if self.opt.random_sample == "patch":
            indx = np.random.randint(0, width - subsamplesize + 1)
            indy = np.random.randint(0, height - subsamplesize + 1)
            px, py = np.meshgrid(
                np.arange(indx, indx + subsamplesize).astype(np.float32),
                np.arange(indy, indy + subsamplesize).astype(np.float32))
        elif self.opt.random_sample == "random":
            px = np.random.randint(0,
                                   width,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.randint(0,
                                   height,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "random2":
            px = np.random.uniform(0,
                                   width - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
            py = np.random.uniform(0,
                                   height - 1e-5,
                                   size=(subsamplesize,
                                         subsamplesize)).astype(np.float32)
        elif self.opt.random_sample == "proportional_random":
            px, py = self.proportional_select(gt_mask)
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        raydir = get_blender_raydir(pixelcoords, self.height, self.width, focal, camrot, self.opt.dir_norm > 0)
        item["pixel_idx"] = pixelcoords
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)

        for key, value in item.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            item[key] = value.unsqueeze(0)

        return item

