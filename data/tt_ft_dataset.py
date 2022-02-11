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
from plyfile import PlyData, PlyElement
import copy
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import h5py

from data.base_dataset import BaseDataset
import configparser

from os.path import join
import cv2
# import torch.nn.functional as F
from .data_utils import get_dtu_raydir

FLIP_Z = np.asarray([
    [1,0,0],
    [0,1,0],
    [0,0,-1],
], dtype=np.float32)

def colorjitter(img, factor):
    # brightness_factor,contrast_factor,saturation_factor,hue_factor
    # img = F.adjust_brightness(img, factor[0])
    # img = F.adjust_contrast(img, factor[1])
    img = F.adjust_saturation(img, factor[2])
    img = F.adjust_hue(img, factor[3]-1.0)

    return img


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    # c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = rot_beta(theta/180.*np.pi) @ c2w
    # c2w = rot_beta(90/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    c2w = c2w #@ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return c2w

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

rot_beta = lambda th : np.asarray([
    [np.cos(th),-np.sin(th), 0, 0],
    [np.sin(th),np.cos(th), 0, 0],
    [0,0,1,0],
    [0,0,0,1],
], dtype=np.float32)

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
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions

class TtFtDataset(BaseDataset):

    def initialize(self, opt, img_wh=[1920,1080], downSample=1.0, max_len=-1, norm_w2c=None, norm_c2w=None):
        self.opt = opt
        self.data_dir = opt.data_root
        self.scan = opt.scan
        self.split = opt.split

        self.img_wh = (int(opt.img_wh[0] * downSample), int(opt.img_wh[1] * downSample))
        self.downSample = downSample
        self.alphas=None
        self.scale_factor = 1.0 / 1.0
        self.max_len = max_len
        # self.cam_trans = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
        self.cam_trans = np.diag(np.array([-1, 1, 1, 1], dtype=np.float32))
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
        self.build_init_metas()
        self.norm_w2c, self.norm_c2w = torch.eye(4, device="cuda", dtype=torch.float32), torch.eye(4, device="cuda", dtype=torch.float32)
        self.near_far = np.array([opt.near_plane, opt.far_plane])

        self.intrinsic = self.get_instrinsic()
        img = Image.open(self.image_paths[0])
        self.ori_img_shape = list(self.transform(img).shape)  # (4, h, w)
        self.intrinsic[0, :] *= (self.width / self.ori_img_shape[2])
        self.intrinsic[1, :] *= (self.height / self.ori_img_shape[1])
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats()


        if self.split != "render":
            self.build_init_view_lst()
            self.total = len(self.id_list)
            print("dataset total:", self.split, self.total)
        else:
            self.get_render_poses()
            print("render only, pose total:", self.total)


    def get_render_poses(self):
        # print("pose file", os.path.join(self.data_dir, self.scan, "test_traj.txt"))
        # self.render_poses = np.loadtxt(os.path.join(self.data_dir, self.scan, "test_traj.txt")).reshape(-1,4,4)
        # print("self.render_poses", self.render_poses)
        # self.total = len(self.render_poses)

        stride = 100  # self.opt.render_stride
        # radius = 1.6  # self.opt.render_radius  @ self.blender2opencv
        parameters = {"Ignatius": [1.7, 1.7, -87.0], "Truck": [2.5, 1.5, 91.0],
                      "Caterpillar": [2.2, 2.2, -89.0], "Family": [0.9, 0.9, -91.0]
                    , "Barn": [2.5, 2.5, 88.0]}
        a, b, phi = parameters[self.opt.scan]  # self.opt.render_radius  @ self.blender2opencv

        self.render_poses = np.stack([pose_spherical(angle, phi, self.radius_func(angle, a, b)) @ self.blender2opencv for angle in np.linspace(-180, 180, stride + 1)[:-1]], 0)
        # print("self.render_poses", self.render_poses[0])
        self.total = len(self.render_poses)

    def radius_func(self, angle, a, b):
        # return 1.2 + abs(np.cos((180 + angle - 36) * np.pi / 180) * radius)
        theta = (angle - (36-180)) * np.pi / 180
        return a * b / np.sqrt(a*a*np.sin(theta)**2 + b*b*np.cos(theta)**2)

    def get_instrinsic(self):
        filepath = os.path.join(self.data_dir, self.scan, "intrinsics.txt")
        try:
            intrinsic = np.loadtxt(filepath).astype(np.float32)[:3, :3]
            return intrinsic
        except ValueError:
            pass

            # Get camera intrinsics
        with open(filepath, 'r') as file:
            f, cx, cy, _ = map(float, file.readline().split())
        fy=fx = f

        # Build the intrinsic matrices
        intrinsic = np.array([[fx, 0., cx],
                                   [0., fy, cy],
                                   [0., 0, 1]])
        return intrinsic



    @staticmethod
    def modify_commandline_options(parser, is_train):
        # ['random', 'random2', 'patch'], default: no random samplec
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
        parser.add_argument(
            '--img_wh',
            type=int,
            nargs=2,
            default=(1920, 1080),
            # default=(1088, 640),
            help='resize target of the image'
        )
        parser.add_argument(
            '--mvs_img_wh',
            type=int,
            nargs=2,
            # default=(1920, 1080), 1590, 960
            default=(1088, 640),
            help='resize target of the image'
        )
        return parser


    def build_init_metas(self):
        colordir = os.path.join(self.data_dir, self.scan, "rgb")
        train_image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f)) and f.startswith("0")]
        test_image_paths = [f for f in os.listdir(colordir) if os.path.isfile(os.path.join(colordir, f)) and f.startswith("1")]
        self.train_id_list = list(range(len(train_image_paths)))
        self.test_id_list = list(range(len(test_image_paths)))
        self.train_image_paths = ["" for i in self.train_id_list]
        self.test_image_paths = ["" for i in self.test_id_list]
        self.train_pos_paths = ["" for i in self.train_id_list]
        self.test_pos_paths = ["" for i in self.test_id_list]
        for train_path in train_image_paths:
            id = int(train_path.split("_")[1])
            self.train_image_paths[id] = os.path.join(self.data_dir, self.scan, "rgb/{}".format(train_path))
            self.train_pos_paths[id] = os.path.join(self.data_dir, self.scan, "pose/{}.txt".format(train_path[:-4]))
        for test_path in test_image_paths:
            id = int(test_path.split("_")[1])
            self.test_image_paths[id] = os.path.join(self.data_dir, self.scan, "rgb/{}".format(test_path))
            self.test_pos_paths[id] = os.path.join(self.data_dir, self.scan, "pose/{}.txt".format(test_path[:-4]))
        self.id_list = self.train_id_list if self.split=="train" else self.test_id_list
        self.pos_paths = self.train_pos_paths if self.split=="train" else self.test_pos_paths
        self.image_paths = self.train_image_paths if self.split=="train" else self.test_image_paths
        if self.opt.ranges[0] > -90.0:
            self.spacemin, self.spacemax = torch.as_tensor(self.opt.ranges[:3]), torch.as_tensor(self.opt.ranges[3:6])
        else:
            minmax = np.loadtxt(os.path.join(self.data_dir, self.scan, "bbox.txt")).astype(np.float32)[:6]
            self.spacemin, self.spacemax = torch.as_tensor(minmax[:3]), torch.as_tensor(minmax[3:6])

    def build_init_view_lst(self):
        self.view_id_list = []
        cam_xyz_lst = [c2w[:3,3] for c2w in self.cam2worlds]
        # _, _, w2cs, c2ws = self.build_proj_mats(meta=self.testmeta, list=self.test_id_list)
        # test_cam_xyz_lst = [c2w[:3,3] for c2w in c2ws]
        cam_points = [np.array([[0, 0, 0.1]], dtype=np.float32) @ c2w[:3, :3].T for c2w in self.cam2worlds]
        if self.split=="train":
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            cam_points = np.concatenate(cam_points, axis=0) + cam_xyz
            # test_cam_xyz = np.stack(test_cam_xyz_lst, axis=0)
            print("cam_points", cam_points.shape, cam_xyz.shape, np.linalg.norm(cam_xyz, axis=-1))
            triangles = data_utils.triangluation_bpa(cam_xyz, test_pnts=cam_points, full_comb=self.opt.full_comb>0)
            self.view_id_list = [triangles[i] for i in range(len(triangles))]


    def define_transforms(self):
        self.transform = T.ToTensor()


    def build_proj_mats(self):
        proj_mats, world2cams, cam2worlds, intrinsics = [], [], [], []
        list = self.id_list
        dintrinsic = self.get_instrinsic()
        dintrinsic[0, :] *= (self.opt.mvs_img_wh[0] / self.ori_img_shape[2])
        dintrinsic[1, :] *= (self.opt.mvs_img_wh[1] / self.ori_img_shape[1])

        for vid in list:
            c2w = np.loadtxt(self.pos_paths[vid]) # @ self.cam_trans
            w2c = np.linalg.inv(c2w)
            cam2worlds.append(c2w)
            world2cams.append(w2c)
            intrinsics.append(dintrinsic)
            proj_mat_l = np.eye(4)
            downintrinsic = copy.deepcopy(dintrinsic)
            downintrinsic[:2] = downintrinsic[:2] / 4
            proj_mat_l[:3, :4] = downintrinsic @ w2c[:3, :4]
            proj_mats += [(proj_mat_l, self.near_far)]

        proj_mats = np.stack(proj_mats)
        intrinsics = np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds



    def define_transforms(self):
        self.transform = T.ToTensor()


    def read_meta(self):

        w, h = self.img_wh
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.blackimgs = []
        self.whiteimgs = []
        self.depths = []
        self.alphas = []

        self.view_id_dict = {}
        self.directions = get_ray_directions(h, w, [self.focal, self.focal])  # (h, w, 3)

        count = 0
        for i, idx in enumerate(self.id_list):
            frame = self.meta['frames'][idx]

            image_path = os.path.join(self.data_dir, self.scan, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            self.depths += [(img[-1:, ...] > 0.1).numpy().astype(np.float32)]
            self.alphas += [img[-1:].numpy().astype(np.float32)]
            self.blackimgs += [img[:3] * img[-1:]]
            self.whiteimgs += [img[:3] * img[-1:] + (1 - img[-1:])]


            # ray directions for all pixels, same for all images (same H, W, focal)

            # rays_o, rays_d = get_rays(self.directions, self.cam2worlds[i])  # both (h*w, 3)
            #
            # self.all_rays += [torch.cat([rays_o, rays_d,
            #                              self.near_far[0] * torch.ones_like(rays_o[:, :1]),
            #                              self.near_far[1] * torch.ones_like(rays_o[:, :1])], 1)]  # (h*w, 8)
            self.view_id_dict[idx] = i
        self.poses = self.cam2worlds


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


    def read_img_path(self, image_path, img_wh, black=False):
        img = Image.open(image_path)
        img = img.resize(img_wh, Image.LANCZOS)
        img = self.transform(img)  # (4, h, w)

        if img.shape[0] == 4:
            alpha = img[-1:].numpy().astype(np.float32)
            blackimg = img[:3] * img[-1:]
            whiteimg = img[:3] * img[-1:] + (1 - img[-1:])
            return blackimg, whiteimg, alpha[0,...] > 0

        # print("img",img)
        alpha = torch.norm(1.0 - img, dim=0) > 0.0001
        blackimg = None
        if black:
            blackimg = img[:3] * alpha[None, ...]

        # print("alpha", torch.sum(alpha))
        return blackimg, img, alpha

    def get_init_alpha(self):
        self.alphas = []
        for i in self.id_list:
            vid = i
            _, _, alpha = self.read_img_path(self.image_paths[vid], self.opt.mvs_img_wh)
            self.alphas += [alpha[None, ...]]
        # self.alphas = np.stack(self.alphas).astype(np.float32)  # (V, H, W)

    def get_init_item(self, idx, crop=False):
        if self.alphas is None:
            self.get_init_alpha()
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        mvs_images, imgs, depths_h, alphas = [], [], [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = i
            blackimg, img, alpha = self.read_img_path(self.image_paths[vid], self.opt.mvs_img_wh, black=True)
            mvs_images += [blackimg]
            alphas+= [alpha]
            imgs += [img]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            near_fars.append(near_far)
            # print("idx",idx, vid, self.image_paths[vid])
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
        mvs_images = np.stack(mvs_images).astype(np.float32)

        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        # view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        # c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = mvs_images  # (V, 3, H, W)
        # sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['alphas'] = np.stack(alphas).astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars_depth'] = near_fars.astype(np.float32)[0]
        sample['near_fars'] = np.tile(self.near_far.astype(np.float32)[None,...],(len(near_fars),1))
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



    def __getitem__(self, id, crop=False, full_img=False):
        item = {}
        _, img, _ = self.read_img_path(self.image_paths[id], self.img_wh)
        w2c = self.world2cams[id]
        c2w = self.cam2worlds[id]
        intrinsic = self.intrinsic
        _, near_far = self.proj_mats[id]

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
        item["c2w"] = torch.from_numpy(c2w).float()
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([near_far[1]]).view(1, 1)
        item['near'] = torch.FloatTensor([near_far[0]]).view(1, 1)
        item['h'] = height
        item['w'] = width
        # item['depths_h'] = self.depths[id]
        # bounding box
        if full_img:
            item['images'] = img[None,...]
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
        item['id'] = id

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



    def get_item(self, idx, crop=False, full_img=False):
        item = self.__getitem__(idx, crop=crop, full_img=full_img)

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
        camrot = transform_matrix[0:3, 0:3]
        campos = transform_matrix[0:3, 3]
        # focal = self.focal

        # item["focal"] = focal
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float()
        item['lightpos'] = item["campos"]
        item['intrinsic'] = self.intrinsic

        # near far
        item['far'] = torch.FloatTensor([self.opt.far_plane]).view(1, 1)
        item['near'] = torch.FloatTensor([self.opt.near_plane]).view(1, 1)
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
            raise Exception("no gt_mask, no proportional_random !!!")
        else:
            px, py = np.meshgrid(
                np.arange(width).astype(np.float32),
                np.arange(height).astype(np.float32))

        pixelcoords = np.stack((px, py), axis=-1).astype(np.float32)  # H x W x 2
        # raydir = get_cv_raydir(pixelcoords, self.height, self.width, focal, camrot)
        item["pixel_idx"] = pixelcoords
        # print("pixelcoords", pixelcoords.reshape(-1,2)[:10,:])
        raydir = get_dtu_raydir(pixelcoords, self.intrinsic, camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        item['id'] = idx

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

