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
import itertools
import random
from torch.utils.data import Dataset, DataLoader
import torch
import os
from PIL import Image
import h5py
from . import data_utils
from utils import util
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

class DtuFtDataset(BaseDataset):

    def initialize(self, opt, n_views=3, img_wh=[640,512], downSample=1.0, max_len=-1, norm_w2c=None, norm_c2w=None):
        self.opt = opt
        self.data_dir = opt.data_root
        self.scan = opt.scan
        self.split = opt.split

        assert int(640 * downSample) % 32 == 0, \
            f'image width is {int(640 * downsample)}, it should be divisible by 32, you may need to modify the imgScale'
        self.img_wh = (int(640 * downSample), int(512 * downSample))
        self.downSample = downSample

        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.n_views = n_views
        self.define_transforms()

        self.pair_idx = torch.load('../data/dtu_configs/pairs.th')
        self.pair_idx = [self.pair_idx['dtu_train'],self.pair_idx['dtu_test']]
        print("dtu_ft train id", self.pair_idx[0])
        print("dtu_ft test id", self.pair_idx[1])
        self.bbox_3d = torch.tensor([[-1.0, -1.0, 2.2], [1.0, 1.0, 4.2]])
        # self.near_far = np.asarray([2.125, 4.525])

        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'

            self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        if not self.opt.bg_color or self.opt.bg_color == 'black':
            self.bg_color = (0, 0, 0)
        elif self.opt.bg_color == 'white':
            self.bg_color = (1, 1, 1)
        elif self.opt.bg_color == 'random':
            self.bg_color = 'random'
        else:
            self.bg_color = [float(one) for one in self.opt.bg_color.split(",")]


        self.build_init_metas()
        self.norm_w2c, self.norm_c2w = torch.eye(4, device="cuda", dtype=torch.float32), torch.eye(4, device="cuda", dtype=torch.float32)
        if opt.normview > 0:
            _, _ , w2cs, c2ws = self.build_proj_mats(list=self.pair_idx[1])
            norm_w2c, norm_c2w = self.normalize_cam(w2cs, c2ws)
        if opt.normview == 2:
            self.norm_w2c, self.norm_c2w = torch.as_tensor(norm_w2c, device="cuda", dtype=torch.float32), torch.as_tensor(norm_c2w, device="cuda", dtype=torch.float32)
            norm_w2c, norm_c2w = None, None
        self.proj_mats, self.intrinsics, self.world2cams, self.cam2worlds = self.build_proj_mats(norm_w2c=norm_w2c, norm_c2w=norm_c2w)

        self.build_view_lst()
        if opt.split != "render":
            self.read_meta()
            self.total = len(self.id_list)
        else:
            self.get_render_poses()
            self.total = len(self.render_poses)
        print("dataset total:", self.split, self.total)

    def get_render_poses(self):
        self.render_poses = util.gen_render_path(self.cam2worlds[:3,...], N_views=60)

        # cam_xyz_lst = [c2w[:3, 3] for c2w in self.cam2worlds]
        # cam_xyz = np.stack(cam_xyz_lst, axis=0)
        # triangles = data_utils.triangluation_bpa(cam_xyz, test_pnts=None, full_comb=False)
        # self.render_poses = util.gen_render_path_contour(triangles, self.cam2worlds, N_views=200)

    # def gen_render_path(c2ws, N_views=30):
    #     N = len(c2ws)
    #     rotvec, positions = [], []
    #     rotvec_inteplat, positions_inteplat = [], []
    #     weight = np.linspace(1.0, .0, N_views // 3, endpoint=False).reshape(-1, 1)
    #     for i in range(N):
    #         r = R.from_matrix(c2ws[i, :3, :3])
    #         euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
    #         if i:
    #             mask = np.abs(euler_ange - rotvec[0]) > 180
    #             euler_ange[mask] += 360.0
    #         rotvec.append(euler_ange)
    #         positions.append(c2ws[i, :3, 3:].reshape(1, 3))
    #
    #         if i:
    #             rotvec_inteplat.append(weight * rotvec[i - 1] + (1.0 - weight) * rotvec[i])
    #             positions_inteplat.append(weight * positions[i - 1] + (1.0 - weight) * positions[i])
    #
    #     rotvec_inteplat.append(weight * rotvec[-1] + (1.0 - weight) * rotvec[0])
    #     positions_inteplat.append(weight * positions[-1] + (1.0 - weight) * positions[0])
    #
    #     c2ws_render = []
    #     angles_inteplat, positions_inteplat = np.concatenate(rotvec_inteplat), np.concatenate(positions_inteplat)
    #     for rotvec, position in zip(angles_inteplat, positions_inteplat):
    #         c2w = np.eye(4)
    #         c2w[:3, :3] = R.from_euler('xyz', rotvec, degrees=True).as_matrix()
    #         c2w[:3, 3:] = position.reshape(3, 1)
    #         c2ws_render.append(c2w.copy())
    #     c2ws_render = np.stack(c2ws_render)
    #     return c2ws_render




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
            default=2, #2.125,
            help=
            'Near clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument(
            '--far_plane',
            type=float,
            default=6, #4.525,
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
            '--full_comb',
            type=int,
            default=0,
            help=''
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
        parser.add_argument('--uni_depth',
                            type=int,
                            default=0,
                            help='normalize the ray_dir to unit length or not, default not')

        return parser


    def define_transforms(self):
        self.transform = T.ToTensor()

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = self.opt.near_plane if self.opt.uni_depth > 0 else float(lines[11].split()[0]) * self.scale_factor
        depth_max = self.opt.far_plane if self.opt.uni_depth > 0 else depth_min + float(lines[11].split()[1]) * 192 * 1.06 * self.scale_factor
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]

    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        return depth_h



    def load_poses_all(self):
        c2ws = []
        List = sorted(os.listdir(os.path.join(self.data_dir, f'Cameras/train/')))
        for item in List:
            proj_mat_filename = os.path.join(self.data_dir, f'Cameras/train/{item}')
            intrinsic, w2c, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            c2ws.append(np.linalg.inv(w2c))
        self.focal = [intrinsic[0, 0], intrinsic[1, 1]]
        return np.stack(c2ws)

    def build_view_lst(self):
        cam_xyz_lst = [c2w[:3, 3] for c2w in self.cam2worlds]
        if self.opt.full_comb == 1:
            # pass
            triangles = list(itertools.combinations(self.id_list, 3))
            self.view_id_list = []
            for tris in triangles:
                tris = list(tris)
                random.shuffle(tris)
                self.view_id_list.append(tris)
        elif self.opt.full_comb > 1:
            # pass
            cam_xyz = np.stack(cam_xyz_lst, axis=0)
            # test_cam_xyz = np.stack(test_cam_xyz_lst, axis=0)
            # if self.opt.full_comb <= 1:
            triangles = data_utils.triangluation_bpa(cam_xyz, test_pnts=None, full_comb=True)
            if self.opt.full_comb == 2:
                self.view_id_list = [triangles[i] for i in range(len(triangles))]
            elif self.opt.full_comb in [3, 4]:  # 1 jump
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
                        triplets += [[key] + list(pair) for pair in pairs]
                    self.view_id_list = [triplets[i] for i in range(len(triplets))]
                elif self.opt.full_comb == 4:
                    second_dict = copy.deepcopy(first_dict)
                    for key, val in first_dict.items():
                        for second in val:
                            second_dict[key] += first_dict[second]
                        second_dict[key] = list(unique_lst(second_dict[key]))
                        second_dict[key] = [val for val in second_dict[key] if
                                            val != key and val not in first_dict[key]]
                        # print("key val", key, second_dict[key])
                    for key, val in second_dict.items():
                        pairs = list(itertools.combinations(second_dict[key], 2))
                        print("key val", key, pairs)
                        triplets += [[key] + list(pair) for pair in pairs]
                    print("len()", len(triplets))
                    # exit()
                    self.view_id_list = [triplets[i] for i in range(len(triplets))]
            for i in range(len(self.view_id_list)):
                triplets = self.view_id_list[i]
                real_trip = [self.id_list[j] for j in triplets]
                self.view_id_list[i] = real_trip

    def build_init_metas(self):
        self.view_id_list = []
        self.id_list = []
        if self.split != "test":
            with open(f'../data/dtu_configs/dtu_finetune_init_pairs.txt') as f:
                num_viewpoint = int(f.readline())
                # viewpoints (16)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    str_lst=f.readline().rstrip().split(',')
                    src_views = [int(x) for x in str_lst]
                    self.view_id_list.append([ref_view] + src_views)
                    self.id_list.append(ref_view)
        else:
            self.id_list = self.pair_idx[1]

        if self.split == "comb":
            self.id_list += self.pair_idx[1]




        with open(f'../data/dtu_configs/lists/dtu_test_ground.txt') as f:
            lines = f.readlines()
            for line in lines:
                info = line.strip().split()
                if self.scan == info[0]:
                    self.plane_ind = int(info[1])
                    print("self.plane_ind", self.plane_ind)
                    break

        if self.opt.full_comb < 0:
            with open(f'../data/nerf_synth_configs/list/lego360_init_pairs.txt') as f:
                for line in f:
                    str_lst = line.rstrip().split(',')
                    src_views = [int(x) for x in str_lst]
                    self.view_id_list.append(src_views)


    def build_proj_mats(self, list=None, norm_w2c=None, norm_c2w=None):
        list = self.id_list if list is None else list
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in list:
            proj_mat_filename = os.path.join(self.data_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor

            if norm_c2w is not None:
                extrinsic = extrinsic @ norm_c2w
            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics += [intrinsic.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

            proj_mats += [(proj_mat_l, near_far)]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        proj_mats, intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        world2cams, cam2worlds = np.stack(world2cams), np.stack(cam2worlds)
        return proj_mats, intrinsics, world2cams, cam2worlds


    def bcart2sphere(self, xyz):
        r = np.linalg.norm(xyz, axis=1)
        xyn = np.linalg.norm(xyz[...,:2], axis=1)
        th = np.arctan2(xyn, xyz[...,2])
        ph = np.arctan2(xyz[...,1], xyz[...,0])
        print("r", r.shape, r, xyn.shape, th.shape, ph.shape)
        return np.stack([r,th,ph], axis=-1)


    def sphere2cart(self, rtp):
        r, th, ph = rtp[0], rtp[1], rtp[2]
        x = r * np.sin(th) * np.cos(ph)
        y = r * np.sin(th) * np.sin(ph)
        z = r * np.cos(th)
        return np.asarray([x,y,z])


    def matrix2euler(self, M):
        x = np.arctan2(-M[1][2], M[2][2])
        cosY = np.sqrt(1 - M[0][2])
        y = np.arctan2(M[0][2], cosY)
        sinZ = np.cos(x) * M[1][0] + np.sin(x) * M[2][0]
        cosZ = np.cos(x) * M[1][1] + np.sin(x) * M[2][1]
        z = np.arctan2(sinZ, cosZ)
        return np.asarray([x,y,z])


    def euler2matrix(self, xyz):
        Cxyz = np.cos(xyz)
        Sxyz = np.sin(xyz)

        Cx, Cy, Cz = Cxyz[0], Cxyz[1], Cxyz[2]
        Sx, Sy, Sz = Sxyz[0], Sxyz[1], Sxyz[2]

        M = [[Cy*Cz, -Cy*Sz, Sy],
             [Sx*Sy*Cz + Cx*Sz, -Sx*Sy*Sz + Cx*Cz, -Sx*Cy],
             [-Cx*Sy*Cz + Sx*Sz, Cx*Sy*Sz + Sx*Cz, Cx*Cy]
        ]
        return np.array(M)


    def normalize_cam(self, w2cs, c2ws):
        # cam_xyz = c2ws[..., :3, 3]
        # rtp = self.bcart2sphere(cam_xyz)
        # print(rtp.shape)
        # rtp = np.mean(rtp, axis=0)
        # avg_xyz = self.sphere2cart(rtp)
        # euler_lst = []
        # for i in range(len(c2ws)):
        #     euler_angles = self.matrix2euler(c2ws[i][:3,:3])
        #     print("euler_angles", euler_angles)
        #     euler_lst += [euler_angles]
        # euler = np.mean(np.stack(euler_lst, axis=0), axis=0)
        # print("euler mean ",euler)
        # M = self.euler2matrix(euler)
        # norm_c2w = np.eye(4)
        # norm_c2w[:3,:3] = M
        # norm_c2w[:3,3] = avg_xyz
        # norm_w2c = np.linalg.inv(norm_c2w)
        # return norm_w2c, norm_c2w
        index=0
        return w2cs[index], c2ws[index]

    def read_meta(self):

        # sub select training views from pairing file
        # if os.path.exists('configs/pairs.th'):
        #     self.img_idx = self.pair_idx[0] if 'train'== self.split else self.pair_idx[1]
        #     print(f'===> {self.split}ing index: {self.img_idx}')

        # name = os.path.basename(self.data_dir)
        # test_idx = torch.load('configs/pairs.th')[f'{name}_test']
        # self.img_idx = test_idx if self.split!='train' else np.delete(np.arange(0,49),test_idx)

        w, h = self.img_wh
        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.imgs = []
        self.depths = []
        self.all_rgbs = []
        self.all_depth = []
        self.view_id_dict = {}
        count = 0
        for i, idx in enumerate(self.id_list):
            image_path = os.path.join(self.data_dir,
                                        f'Rectified/{self.scan}_train/rect_{idx + 1:03d}_3_r5000.png')
            depth_filename = os.path.join(self.data_dir,
                                          f'Depths_raw/{self.scan}/depth_map_{idx:04d}.pfm')
            self.image_paths += [image_path]
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)
            self.imgs += [img]
            self.all_rgbs += [img.reshape(3, -1).permute(1, 0)]  # (h*w, 3) RGBA

            if os.path.exists(depth_filename):
                depth = self.read_depth(depth_filename)
                depth *= self.scale_factor
                self.depths += [depth]
                self.all_depth += [torch.from_numpy(depth).float().view(-1,1)]

            # ray directions for all pixels, same for all images (same H, W, focal)
            intrinsic = self.intrinsics[count]
            # center = [intrinsic[0,2], intrinsic[1,2]]
            self.focal = [intrinsic[0,0], intrinsic[1,1]]
            # self.directions = get_ray_directions(h, w, self.focal, center)  # (h, w, 3)
            # rays_o, rays_d = get_rays(self.directions, self.cam2worlds[i])  # both (h*w, 3)
            #
            # self.all_rays += [torch.cat([rays_o, rays_d,
            #                              self.near_far[0] * torch.ones_like(rays_o[:, :1]),
            #                              self.near_far[1] * torch.ones_like(rays_o[:, :1])], 1)]  # (h*w, 8)
            self.view_id_dict[idx] = i
        self.poses = self.cam2worlds
        # if 'train' == self.split:
        #     self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
        #     self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)
        # else:
        #     self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames]),h*w, 3)
        #     self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        #     self.all_depth = torch.stack(self.all_depth, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames]),h,w,3)
        #     count+=1


    def __len__(self):
        return self.total

    def name(self):
        return 'DtuDataset'


    def __del__(self):
        print("end loading")

    def get_campos_ray(self):
        centerpixel = np.asarray(self.img_wh).astype(np.float32)[None, :] // 2
        camposes = []
        centerdirs = []
        for i, idx in enumerate(self.id_list):
            c2w = self.cam2worlds[i].astype(np.float32)
            campos = c2w[:3, 3]
            camrot = c2w[:3, :3]
            raydir = get_dtu_raydir(centerpixel, self.intrinsics[0].astype(np.float32), camrot, True)
            camposes.append(campos)
            centerdirs.append(raydir)
        camposes = np.stack(camposes, axis=0)  # 2091, 3
        centerdirs = np.concatenate(centerdirs, axis=0)  # 2091, 3
        # print("camposes", camposes.shape, centerdirs.shape)
        return torch.as_tensor(camposes, device="cuda", dtype=torch.float32), torch.as_tensor(centerdirs, device="cuda",
                                                                                              dtype=torch.float32)


    def get_init_item(self, idx, crop=False):
        sample = {}
        init_view_num = self.opt.init_view_num
        view_ids = self.view_id_list[idx]
        if self.split == 'train':
            view_ids = view_ids[:init_view_num]

        affine_mat, affine_mat_inv = [], []
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i in view_ids:
            vid = self.view_id_dict[i]
            imgs += [self.imgs[vid]]
            proj_mat_ls, near_far = self.proj_mats[vid]
            intrinsics.append(self.intrinsics[vid])
            w2cs.append(self.world2cams[vid])
            c2ws.append(self.cam2worlds[vid])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            depths_h.append(self.depths[vid])
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

        depths_h = np.stack(depths_h)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        # view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        # c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = imgs #self.normalize_rgb(imgs)  # (V, 3, H, W)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars_depth'] = near_fars.astype(np.float32)[0]
        sample['near_fars'] = np.tile(sample['near_fars_depth'][None,...],(len(imgs),1))  #np.tile(self.near_far.astype(np.float32)[None,...],(len(near_fars),1))
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


    def normalize_rgb(self, data):
        # to unnormalize image for visualization
        # data V, C, H, W
        V, C, H, W = data.shape
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        return (data - mean) / std



    def __getitem__(self, id, crop=False, full_img=False):
        item = {}
        img = self.imgs[id]
        if full_img:
            item['images'] = img[None,...]
        w2c = self.world2cams[id]
        c2w = self.cam2worlds[id]
        intrinsic = self.intrinsics[id]
        proj_mat_ls, near_far = self.proj_mats[id]

        gt_image = np.transpose(img, (1,2,0))
        # print("gt_image", gt_image.shape)
        width, height = gt_image.shape[1], gt_image.shape[0]
        camrot = (c2w[0:3, 0:3])
        campos = c2w[0:3, 3]
        # print("camrot", camrot, campos)
        item["c2w"] = torch.from_numpy(c2w).float()

        item["intrinsic"] = intrinsic
        # item["intrinsic"] = sample['intrinsics'][0, ...]
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([near_far[1]]).view(1, 1)
        item['near'] = torch.FloatTensor([near_far[0]]).view(1, 1)
        item['h'] = height
        item['w'] = width

        plane_pnt, plane_normal, plane_color = self.get_plane_param(self.plane_ind)
        item['plane_pnt'] = torch.FloatTensor(plane_pnt)
        item['plane_normal'] = torch.FloatTensor(plane_normal)
        item['plane_color'] = torch.FloatTensor(plane_color)
        item['depths_h'] = self.depths[id]
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
        item['id'] = id
        item['vid'] = self.id_list[id]

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

        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float()
        item['lightpos'] = item["campos"]
        item['intrinsic'] = self.intrinsics[0]

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
        raydir = get_dtu_raydir(pixelcoords, self.intrinsics[0], camrot, self.opt.dir_norm > 0)
        raydir = np.reshape(raydir, (-1, 3))
        item['raydir'] = torch.from_numpy(raydir).float()
        item['id'] = idx

        plane_pnt, plane_normal, plane_color = self.get_plane_param(self.plane_ind)
        item['plane_pnt'] = torch.FloatTensor(plane_pnt)
        item['plane_normal'] = torch.FloatTensor(plane_normal)
        item['plane_color'] = torch.FloatTensor(plane_color)
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

    def get_plane_param(self, ind):
        plane_pnt = [[-0.49666997, 0.52160616, 3.6239593], [0.20770223, -0.74818161,  3.98697683], [-0.04889537, -0.84123057, 4.03164617]][ind]
        plane_normal = [[-0.11364093, 0.38778102, 0.91471942], [-0.11165793, 0.3806543, 0.91795142], [-0.11154823,  0.3783277, 0.91892608]][ind]

        plane_color = [[1.0, 1.0, 1.0], [150.72447808/255, 99.68367002/255, 63.40976961/255], [80.28243032/255, 54.3915082/255, 35.07029825/255]][ind]
        return plane_pnt, plane_normal, plane_color


    def get_plane_param_points(self):
        r, amount = 10, int(8e3)
        plane_pnt, plane_normal, _ = self.get_plane_param(self.plane_ind)
        a,b,c = plane_normal[0], plane_normal[1], plane_normal[2]
        x0,y0,z0=plane_pnt[0],plane_pnt[1],plane_pnt[2],
        x = r * (np.random.rand(amount, 1) - 0.7)
        y = r * (np.random.rand(amount, 1) - 0.6)
        xy = np.concatenate([x,y], axis=-1)
        z = (a*(xy[...,0]-x0) + b*(xy[...,1]-y0))/(-c) + z0
        gen_pnts = torch.as_tensor(np.stack([xy[...,0], xy[...,1], z], axis=-1), device="cuda", dtype=torch.float32)
        featuredim=self.opt.point_features_dim
        if "0" in list(self.opt.point_dir_mode):
            featuredim -= 3
        if "0" in list(self.opt.point_conf_mode):
            featuredim -= 1
        if "0" in list(self.opt.point_color_mode):
            featuredim -= 3
        gen_embedding = torch.rand(1, len(gen_pnts), featuredim, device="cuda", dtype=torch.float32)
        gen_dir = torch.rand(1, len(gen_pnts), 3, device="cuda", dtype=torch.float32)
        gen_dir = gen_dir / torch.clamp(torch.norm(gen_dir, dim=-1, keepdim=True), min=1e-6)
        gen_color = torch.zeros([1, len(gen_pnts), 3], device="cuda", dtype=torch.float32)
        gen_conf = torch.full([1, len(gen_pnts), 1], 0.3, device="cuda", dtype=torch.float32)
        return gen_pnts, gen_embedding, gen_dir, gen_color, gen_conf


    def filter_plane(self, add_xyz):
        thresh = 0.2
        plane_pnt, plane_normal, _ = self.get_plane_param(self.plane_ind)
        a, b, c = plane_normal[0], plane_normal[1], plane_normal[2]
        x0, y0, z0 = plane_pnt[0], plane_pnt[1], plane_pnt[2]
        d = -a * x0 - b * y0 - c * z0
        dist = torch.abs(add_xyz[...,0] * a + add_xyz[...,1] * b + add_xyz[...,2] * c + d)
        return dist < thresh
