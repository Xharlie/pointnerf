from models.mvs.mvs_utils import read_pfm
import os
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F



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


class DtuDataset(BaseDataset):

    def initialize(self, opt, n_views=3, levels=1, img_wh=[640,512], downSample=1.0, max_len=-1):
        self.opt = opt
        self.data_dir = opt.data_root
        if not self.opt.bg_color or self.opt.bg_color == 'black':
            self.bg_color = (0, 0, 0)
        elif self.opt.bg_color == 'white':
            self.bg_color = (1, 1, 1)
        elif self.opt.bg_color == 'random':
            self.bg_color = 'random'
        else:
            self.bg_color = [float(one) for one in self.bg_color.split()]
            if len(self.bg_color) != 3:
                self.bg_color = None

        self.img_wh = img_wh
        self.downSample = downSample
        self.scale_factor = 1.0 / 200
        self.max_len = max_len
        self.n_views = n_views
        self.levels = levels  # FPN levels
        self.split = opt.split
        self.build_metas()
        self.build_proj_mats()
        self.define_transforms()
        self.near_far = np.asarray([2.125, 4.525])
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
            self.height, self.width = int(self.img_wh[1]), int(self.img_wh[0])

        if os.path.isfile(self.data_dir + "/bb.txt"):
            self.bb = np.loadtxt(self.data_dir + "/bb.txt")
            print("boundingbox", self.bb)
        else:
            self.bb = np.array([-1, -1, -1, 1, 1, 1]).reshape(
                (2, 3)).astype(np.float32)

        self.total = len(self.metas)
        print("dataset total:", self.split, self.total)
        return

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
        parser.add_argument('--shape_id', type=int, default=0, help='shape id')
        parser.add_argument('--trgt_id', type=int, default=0, help='shape id')
        parser.add_argument('--num_nn',
                            type=int,
                            default=1,
                            help='number of nearest views in a batch')
        parser.add_argument(
            '--near_plane',
            type=float,
            default=2.0,
            help=
            'Near clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument(
            '--far_plane',
            type=float,
            default=6.0,
            help=
            'Far clipping plane, by default it is computed according to the distance of the camera '
        )
        parser.add_argument('--init_view_num',
                            type=int,
                            default=3,
                            help='number of random samples')
        parser.add_argument(
            '--bg_color',
            type=str,
            default="white",
            help=
            'background color, white|black(None)|random|rgb (float, float, float)'
        )

        # parser.add_argument(
        #     '--z_dir',
        #     type=str,
        #     default="down",
        #     help=
        #     'z axis up (in nerf json), down (in reflectance ply)'
        # )

        parser.add_argument('--inverse_gamma_image',
                            type=int,
                            default=-1,
                            help='de-gamma correct the input image')
        parser.add_argument('--pin_data_in_memory',
                            type=int,
                            default=-1,
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

        return parser


    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    # T.Normalize(mean=[0.485, 0.456, 0.406],
                                    #             std=[0.229, 0.224, 0.225]),
                                    ])

    def build_metas(self):
        self.metas = []
        with open(f'../data/dtu_configs/lists/dtu_{self.split}_all.txt') as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        # light conditions 0-6 for training
        # light condition 3 for testing (the brightest?)
        light_idxs = [3] if 'train' != self.split else range(7)

        self.id_list = []

        for scan in self.scans:
            with open(f'../data/dtu_configs/dtu_pairs.txt') as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    for light_idx in light_idxs:
                        self.metas += [(scan, light_idx, ref_view, src_views)]
                        self.id_list.append([ref_view] + src_views)

        self.id_list = np.unique(self.id_list)
        self.build_remap()

    def build_proj_mats(self):
        proj_mats, intrinsics, world2cams, cam2worlds = [], [], [], []
        for vid in self.id_list:
            proj_mat_filename = os.path.join(self.data_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4
            extrinsic[:3, 3] *= self.scale_factor

            intrinsic[:2] = intrinsic[:2] * self.downSample
            intrinsics += [intrinsic.copy()]

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat_l = np.eye(4)
            intrinsic[:2] = intrinsic[:2] / 4
            proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

            proj_mats += [(proj_mat_l, near_far)]
            world2cams += [extrinsic]
            cam2worlds += [np.linalg.inv(extrinsic)]

        self.proj_mats, self.intrinsics = np.stack(proj_mats), np.stack(intrinsics)
        self.world2cams, self.cam2worlds = np.stack(world2cams), np.stack(cam2worlds)


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
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor * 1.06
        self.depth_interval = float(lines[11].split()[1])
        return intrinsics, extrinsics, [depth_min, depth_max]


    def check_read_depth(self, depth_filename, processed_filename):
        depth_h = np.array(read_pfm(depth_filename)[0], dtype=np.float32)  # (800, 800) ? (1200. 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=self.downSample, fy=self.downSample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        depth_pro = np.array(read_pfm(processed_filename)[0], dtype=np.float32)  # (800, 800) ? (1200. 1600)
        print("depth", depth.shape, depth_pro.shape, np.sum(np.abs(depth-depth_pro)))


    def read_depth(self, filename, downSample=None):
        downSample = self.downSample if downSample is None else downSample
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800) ? (1200. 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        depth_h = cv2.resize(depth_h, None, fx=downSample, fy=downSample,
                             interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        depth = cv2.resize(depth_h, None, fx=1.0 / 4, fy=1.0 / 4,
                           interpolation=cv2.INTER_NEAREST)  # !!!!!!!!!!!!!!!!!!!!!!!!!
        mask = depth > 0
        return depth, mask, depth_h

    def build_remap(self):
        self.remap = np.zeros(np.max(self.id_list) + 1).astype('int')
        for i, item in enumerate(self.id_list):
            self.remap[item] = i

    def __len__(self):
        return len(self.metas) if self.max_len <= 0 else self.max_len


    def name(self):
        return 'DtuDataset'


    def __del__(self):
        print("end loading")


    def __getitem__(self, idx, crop=False):
        sample = {}
        scan, light_idx, target_view, src_views = self.metas[idx]
        if self.split=='train':
            ids = torch.randperm(5)[:3]
            view_ids = [src_views[i] for i in ids] + [target_view]
        else:
            view_ids = [src_views[i] for i in range(3)] + [target_view]


        affine_mat, affine_mat_inv = [], []
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i, vid in enumerate(view_ids):

            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.data_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            # print("img_filename",img_filename)
            depth_filename = os.path.join(self.data_dir,
                                          f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')

            img = Image.open(img_filename)
            # print("img_filename", img_filename, depth_filename)
            img_wh = np.round(np.array(img.size) * self.downSample).astype('int')
            img = img.resize(img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]

            index_mat = self.remap[vid]
            proj_mat_ls, near_far = self.proj_mats[index_mat]
            intrinsics.append(self.intrinsics[index_mat])
            w2cs.append(self.world2cams[index_mat])
            c2ws.append(self.cam2worlds[index_mat])

            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))

            if os.path.exists(depth_filename):
                depth, mask, depth_h = self.read_depth(depth_filename)
                # self.check_read_depth(depth_filename, os.path.join(self.data_dir, f'Depths/{scan}_train/depth_map_{vid:04d}.pfm'))
                depth_h *= self.scale_factor
                depths_h.append(depth_h)
            else:
                depths_h.append(np.zeros((1, 1)))

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
        # if self.split == 'train':
        #     imgs = colorjitter(imgs, 1.0+(torch.rand((4,))*2-1.0)*0.5)
        # imgs = F.normalize(imgs,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        depths_h = np.stack(depths_h)
        # print("proj_mats", proj_mats[0].shape)
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        view_ids_all = [target_view] + list(src_views) if type(src_views[0]) is not list else [j for sub in src_views for j in sub]
        c2ws_all = self.cam2worlds[self.remap[view_ids_all]]

        sample['images'] = imgs  # (V, 3, H, W)
        sample['mvs_images'] = imgs #self.normalize_rgb(imgs)  # (V, 3, H, W)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars_depth'] = near_fars.astype(np.float32)[0]
        sample['near_fars'] = np.tile(self.near_far.astype(np.float32)[None,...],(len(near_fars),1))
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['view_ids'] = np.array(view_ids)
        sample['light_id'] = np.array(light_idx)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        sample['scan'] = scan
        sample['c2ws_all'] = c2ws_all.astype(np.float32)



        item = {}
        gt_image = np.transpose(imgs[self.opt.trgt_id, ...], (1,2,0))
        width, height = gt_image.shape[1], gt_image.shape[0]
        # gt_mask = (gt_image[..., -1] > 0.1).astype(np.float32)
        # item['gt_mask'] = torch.from_numpy(gt_mask).permute(2, 0, 1).float()

        # gt_image = gt_image / 255.0 # already / 255 for blender

        transform_matrix = w2cs[self.opt.ref_vid] @ c2ws[self.opt.trgt_id]
        # transform_matrix = w2cs[0] @ c2ws[0]
        camrot = (transform_matrix[0:3, 0:3])
        campos = transform_matrix[0:3, 3]

        item["intrinsic"] = sample['intrinsics'][self.opt.trgt_id, ...]
        # item["intrinsic"] = sample['intrinsics'][0, ...]
        item["campos"] = torch.from_numpy(campos).float()
        item["camrotc2w"] = torch.from_numpy(camrot).float() # @ FLIP_Z
        item['lightpos'] = item["campos"]

        dist = np.linalg.norm(campos)

        middle = dist + 0.7
        item['middle'] = torch.FloatTensor([middle]).view(1, 1)
        item['far'] = torch.FloatTensor([near_fars[self.opt.trgt_id][1]]).view(1, 1)
        item['near'] = torch.FloatTensor([near_fars[self.opt.trgt_id][0]]).view(1, 1)
        item['h'] = height
        item['w'] = width

        # bounding box
        item['bb'] = torch.from_numpy(self.bb).float()

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
        # gt_mask = np.reshape(gt_mask, (-1, 1))

        # if self.opt.bg_color is not None:
        #     gt_image = np.clip(
        #         np.power(
        #             np.power(gt_image, 2.2) +
        #             (1 - gt_mask) * self.opt.bg_color, 1.0 / 2.2), 0, 1)

        # gt_mask[gt_mask > 0] = 1
        item['gt_image'] = gt_image
        # item['gt_image'] = torch.from_numpy(gt_image).float().contiguous()
        # item["gt_mask"] = torch.from_numpy(gt_mask).float().contiguous()

        if self.bg_color:
            if self.bg_color == 'random':
                val = np.random.rand()
                if val > 0.5:
                    item['bg_color'] = torch.FloatTensor([1, 1, 1])
                else:
                    item['bg_color'] = torch.FloatTensor([0, 0, 0])
            else:
                item['bg_color'] = torch.FloatTensor(self.bg_color)

        sample.update(item)
        return sample



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

        # bounding box
        item['bb'] = torch.from_numpy(self.bb).float()

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



    def check_points_range(self):
        import glob
        from os import path
        data_dir='/home/xharlie/user_space/data/nrData/dtu/'
        near_far=[2.125, 4.525]
        W, H = 640, 512
        downSample=1
        self.scale_factor=1/200
        scale_factor=1/200
        all_min_lst, all_max_lst = [], []
        for idx in range(1, 129):
            scan = "scan{}".format(idx)
            obj_min_lst, obj_max_lst = [], []
            for vid in range(49):
                depth_filename = os.path.join(data_dir, f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
                camfilename = os.path.join(data_dir, f'Cameras/train/{vid:08d}_cam.txt')
                if not path.exists(depth_filename) or not path.exists(camfilename):
                    print("depth_filename:", path.exists(depth_filename), "; camfilename ",path.exists(camfilename))
                    break
                _, _, depth_h = self.read_depth(depth_filename, downSample=downSample)
                depth_h *= scale_factor
                mask = np.logical_and(depth_h >= near_far[0], depth_h <= near_far[1]).reshape(-1)

                intrinsic, extrinsic, near_far = self.read_cam_file(camfilename)
                intrinsic[:2] *= 4
                extrinsic[:3, 3] *= scale_factor
                intrinsic[:2] = intrinsic[:2] * downSample
                w2c = extrinsic
                c2w = np.linalg.inv(extrinsic)

                # mask = torch.logical_and(depth_h >= near_far[0], cam_expected_depth <= near_far[1])
                ndc_expected_depth = (depth_h - near_far[0]) / (near_far[1] - near_far[0]) # 512, 640
                valid_z = ndc_expected_depth
                valid_x = np.arange(W, dtype=np.float32) / (W - 1)
                valid_y = np.arange(H, dtype=np.float32) / (H - 1)
                valid_y, valid_x = np.meshgrid(valid_y, valid_x, indexing="ij") # 512, 640; 512, 640
                # B,N,H,W
                ndc_xyz = np.stack([valid_x, valid_y, valid_z], axis=-1) # 512, 640, 3
                cam_xyz = self.ndc_2_cam(ndc_xyz, near_far, intrinsic, W, H)
                w_xyz = np.concatenate([cam_xyz, np.ones_like(cam_xyz[...,0:1])], axis=-1) @ c2w.T # (327680, 4)
                w_xyz = w_xyz[mask,:3]
                xyz_min_np, xyz_max_np = np.min(w_xyz, axis=-2), np.max(w_xyz, axis=-2)
                obj_min_lst.append(xyz_min_np)
                obj_max_lst.append(xyz_max_np)
                max_edge = max(xyz_max_np-xyz_min_np)
                # print("xyz_min_np, xyz_max_np edges,", xyz_min_np, xyz_max_np, xyz_max_np-xyz_min_np)
            if len(obj_min_lst) > 0:
                obj_min = np.min(np.array(obj_min_lst), axis=-2)
                obj_max= np.max(np.array(obj_max_lst), axis=-2)
                all_min_lst.append(obj_min)
                all_max_lst.append(obj_max)
                print(scan, "min", obj_min, "max", obj_max)
        obj_min = np.min(np.array(all_min_lst), axis=-2)
        obj_max = np.max(np.array(all_max_lst), axis=-2)
        print("xyz_min, xyz_max, edges,", obj_min, obj_max, obj_max-obj_min)

    def ndc_2_cam(self, ndc_xyz, near_far, intrinsic, W, H):
        inv_scale = np.array([[W - 1, H - 1]])
        cam_z = ndc_xyz[..., 2:3] * (near_far[1] - near_far[0]) + near_far[0]
        cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
        cam_xyz = np.concatenate([cam_xy, cam_z], axis=-1).reshape(-1,3)
        cam_xyz = cam_xyz @ np.linalg.inv(intrinsic.T)
        return cam_xyz

if __name__ == '__main__':
    db = DtuDataset()
    db.check_points_range()
    #  python -m data.dtu_dataset