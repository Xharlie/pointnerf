import torch
import os
from torch.utils.data import DataLoader
import imageio
# models
from .models import *
from .renderer import *
from .mvs_utils import *
from . import filter_utils
from ..helpers.networks import init_seq

from ..depth_estimators.mvsnet import MVSNet as Ofcl_MVSNet
from torch.optim.lr_scheduler import CosineAnnealingLR
from inplace_abn import InPlaceABN
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import transforms as T

feature_str_lst=['appr_feature_str0', 'appr_feature_str1', 'appr_feature_str2', 'appr_feature_str3']


def premlp_init(opt):
    in_channels = 63
    out_channels = opt.point_features_dim
    blocks = []
    act = getattr(nn, opt.act_type, None)

    for i in range(opt.shading_feature_mlp_layer1):
        blocks.append(nn.Linear(in_channels, out_channels))
        blocks.append(act(inplace=True))
        in_channels = out_channels
    blocks = nn.Sequential(*blocks)
    init_seq(blocks)
    return blocks


class MvsPointsModel(nn.Module):

    def __init__(self, args):
        super(MvsPointsModel, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.idx = 0

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start = create_mvs(args, mvs_mode=self.args.manual_depth_view, depth=args.depth_grid)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_featmvs']
        if args.pre_d_est is not None and self.args.manual_depth_view > 0 :
            self.load_pretrained_d_est(self.MVSNet, args.pre_d_est)
        self.FeatureNet = self.render_kwargs_train['network_2d']
        self.render_kwargs_train.pop('network_featmvs')
        self.render_kwargs_train.pop('network_2d')
        self.render_kwargs_train['NDC_local'] = False
        if self.args.manual_depth_view == -1:
            self.ProbNet = ProbNet(8).to(device)
        if self.args.shading_feature_mlp_layer0 > 0:
            self.premlp = premlp_init(args)
        # self.eval_metric = [0.01, 0.05, 0.1]
        self.sample_func = getattr(self, args.mvs_point_sampler, None)
        self.cnt = 0


    def load_pretrained_d_est(self, model, pre_d_est):
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(pre_d_est))
        state_dict = torch.load(pre_d_est, map_location=lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict['model'].items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument("--mvs_lr", type=float, default=5e-4,
                            help='learning rate')
        parser.add_argument('--pad', type=int, default=24)
        parser.add_argument('--depth_grid', type=int, default=128)
        parser.add_argument('--prob_thresh', type=float, default=0.8)
        parser.add_argument('--dprob_thresh', type=float, default=0.8)
        parser.add_argument('--num_neighbor', type=int, default=1)
        parser.add_argument('--depth_vid', type=str, default="0", help="0123")
        parser.add_argument('--ref_vid', type=int, default=0, help="0, 1, 2, or 3")
        parser.add_argument('--num_each_depth', type=int, default=1)
        parser.add_argument('--depth_conf_thresh', type=float, default=None)
        parser.add_argument('--depth_occ', type=int, default=0)
        parser.add_argument('--manual_depth_view', type=int, default=0, help="-1 for learning probability, 0 for gt, 1 for pretrained MVSNet")
        parser.add_argument('--pre_d_est', type=str, default=None, help="loading pretrained depth estimator")
        parser.add_argument('--manual_std_depth', type=float, default=0)
        parser.add_argument('--far_plane_shift', type=float, default=None)
        parser.add_argument('--mvs_point_sampler', type=str, default="gau_single_sampler")
        parser.add_argument('--appr_feature_str0',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--appr_feature_str1',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--appr_feature_str2',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--appr_feature_str3',
            type=str,
            nargs='+',
            # default=["imgfeat_0_0123", "vol"],
            default=["imgfeat_0_0", "vol"],
            help=
            "which feature_map")
        parser.add_argument('--vox_res', type=int, default=0, help='vox_resolution if > 0')


    def decode_batch(self, batch, idx=list(torch.arange(4))):
        data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
        pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                    'c2ws': data_mvs['c2ws'].squeeze(),'near_fars':data_mvs['near_fars'].squeeze()}
        return data_mvs, pose_ref


    def normalize_rgb(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        # data N V C H W
        device = data.device
        mean = torch.tensor([0.485, 0.456, 0.406]).view(*shape).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(*shape).to(device)
        return (data - mean) / std


    def gau_single_sampler(self, volume_prob, args, ref_intrinsic, near_far, cam_expected_depth=None, ndc_std_depth=None):
        # volume_prob # ([1, 1, 128, 176, 208])
        if cam_expected_depth is None:
            B, C, D, H, W = volume_prob.shape
            v = 1.0 / D
            ndc_depths = torch.linspace(0.5 * v, 1.0 - 0.5 * v, steps=D, device=volume_prob.device)[None, None, :, None, None].expand(1, 1, -1, H, W)
            # B, C, H, W
            ndc_expected_depth = torch.sum(volume_prob * ndc_depths, dim=2)  # ([1, 1, 1, 176, 208])
            ndc_std_depth = torch.sqrt(torch.sum(volume_prob * torch.square(ndc_depths-ndc_expected_depth), dim=2)) #([1, 1, 176, 208])
            mask = self.prob_filter(args.dprob_thresh, args.num_neighbor, volume_prob, ndc_expected_depth, ndc_std_depth)
        else:
            # [1, 1, 512, 640]
            mask = torch.logical_and(cam_expected_depth >= near_far[0], cam_expected_depth <= near_far[1])
            ndc_expected_depth = (cam_expected_depth - near_far[0]) / (near_far[1] - near_far[0])
        sampled_depth = self.sample_by_gau(ndc_expected_depth, ndc_std_depth, args) #([1, 1, 5, 512, 640])
        ndc_xyz, cam_xyz = self.depth2point(sampled_depth, ref_intrinsic, near_far) # 1, 1, 512, 640, 3

        return ndc_xyz, cam_xyz, ndc_expected_depth.shape[-2:], mask


    def sample_by_gau(self, ndc_expected_depth, ndc_std_depth, args):

        B, C, H, W = ndc_expected_depth.shape
        N = args.num_each_depth
        # [1, 5, 1, 176, 208]
        sampled_depth = ndc_std_depth[:,None,...] * torch.normal(mean=torch.zeros((B, N, C, H, W), device="cuda"), std=torch.ones((B, N, C, H, W), device=ndc_expected_depth.device)) + ndc_expected_depth[:,None,...]
        return torch.clamp(sampled_depth, min=0.0, max=1.0)


    def depth2point(self, sampled_depth, ref_intrinsic, near_far):
        B, N, C, H, W = sampled_depth.shape
        valid_z = sampled_depth
        valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
        valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
        valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
        # B,N,H,W
        valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
        valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
        ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
        cam_xyz = ndc_2_cam(ndc_xyz, near_far, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
        return ndc_xyz, cam_xyz


    def prob_filter(self, thresh, num_neighbor, volume_prob, ndc_expected_depth, ndc_std_depth):
        B, C, D, H, W = volume_prob.shape
        ceil_idx = torch.ceil(ndc_expected_depth)
        lower_idx = ceil_idx - num_neighbor // 2 + 1 # B, C, 1, H, W
        # upper_idx = ceil_idx + num_neighbor // 2
        shifts = torch.arange(0, num_neighbor, device=volume_prob.device, dtype=torch.int64)[None, :, None, None]
        idx = torch.clamp(lower_idx.to(torch.int64) + shifts, min=0, max=D-1) # B, num_neighbor, H, W
        select_probs = torch.gather(torch.squeeze(volume_prob, dim=1), 1, idx) # B, num_neighbor, H, W
        sumprobs = torch.sum(select_probs, dim=1, keepdim=True) #([1, 1, 176, 208])
        mask = sumprobs > thresh
        return mask


    def extract_2d(self, img_feats, view_ids, layer_ids, intrinsics, c2ws, w2cs, cam_xyz, HD, WD, cam_vid=0):
        out_feats = []
        colors = []
        for vid in view_ids:
            w2c = w2cs[:,vid,...] if vid != cam_vid else None
            warp = homo_warp_nongrid_occ if self.args.depth_occ > 0 else homo_warp_nongrid
            src_grid, mask, hard_id_xy = warp(c2ws[:,cam_vid,...], w2c, intrinsics[:,vid,...], cam_xyz, HD, WD, tolerate=0.1)

            warped_feats = []
            for lid in layer_ids:
                img_feat = img_feats[lid] # 3, 32, 128, 160
                warped_src_feat = extract_from_2d_grid(img_feat[vid:vid+1, ...], src_grid, mask)
                if lid == 0:
                    colors.append(warped_src_feat)
                else:
                    warped_feats.append(warped_src_feat)
            warped_feats = torch.cat(warped_feats, dim=-1)
            out_feats.append(warped_feats)
        out_feats = torch.cat(out_feats, dim=-1)
        colors = torch.cat(colors, dim=-1) if len(colors) > 0 else None
        return out_feats, colors


    def get_image_features(self, imgs):
        return self.FeatureNet(imgs[:, :self.args.init_view_num])


    def query_embedding(self, HDWD, cam_xyz, photometric_confidence, img_feats, c2ws, w2cs, intrinsics, cam_vid, pointdir_w=False):

        HD, WD = HDWD
        points_embedding = []
        points_dirs = None
        points_conf = None
        points_colors = None
        for feat_str in getattr(self.args, feature_str_lst[cam_vid]):
            if feat_str.startswith("imgfeat"):
                _, view_ids, layer_ids = feat_str.split("_")
                view_ids = [int(a) for a in list(view_ids)]
                layer_ids = [int(a) for a in list(layer_ids)]
                twoD_feats, points_colors = self.extract_2d(img_feats, view_ids, layer_ids, intrinsics, c2ws, w2cs, cam_xyz, HD, WD, cam_vid=cam_vid)
                points_embedding.append(twoD_feats)
            elif feat_str.startswith("dir"):
                _, view_ids = feat_str.split("_")
                view_ids = torch.as_tensor([int(a) for a in list(view_ids)], dtype=torch.int64, device=cam_xyz.device)
                cam_pos_world = c2ws[:, view_ids, :, 3] # B, V, 4
                cam_trans = w2cs[:, cam_vid, ...] # B, 4, 4
                cam_pos_cam = (cam_pos_world @ cam_trans.transpose(1, 2))[...,:3] # B, V, 4
                points_dirs = cam_xyz[:,:, None, :] - cam_pos_cam[:, None, :, :] # B, N, V, 3 in current cam coord
                points_dirs = points_dirs / (torch.linalg.norm(points_dirs, dim=-1, keepdims=True) + 1e-6)  # B, N, V, 3
                points_dirs = points_dirs.view(cam_xyz.shape[0], -1, 3) @ c2ws[:, cam_vid, :3, :3].transpose(1, 2)
                if not pointdir_w:
                    points_dirs = points_dirs @ c2ws[:, self.args.ref_vid, :3, :3].transpose(1, 2) # in ref cam coord
                # print("points_dirs", points_dirs.shape)
                points_dirs = points_dirs.view(cam_xyz.shape[0], cam_xyz.shape[1], -1)
            elif feat_str.startswith("point_conf"):
                if photometric_confidence is None:
                    photometric_confidence = torch.ones_like(points_embedding[0][...,0:1])
                points_conf = photometric_confidence
        points_embedding = torch.cat(points_embedding, dim=-1)
        if self.args.shading_feature_mlp_layer0 > 0:
            points_embedding = self.premlp(torch.cat([points_embedding, points_colors, points_dirs, points_conf], dim=-1))
        return points_embedding, points_colors, points_dirs, points_conf


    def gen_points(self, batch):
        if 'scan' in batch.keys():
            batch.pop('scan')
        log, loss = {},0
        data_mvs, pose_ref = self.decode_batch(batch)
        imgs, proj_mats = data_mvs['images'], data_mvs['proj_mats']
        near_fars, depths_h = data_mvs['near_fars'], data_mvs['depths_h'] if 'depths_h' in data_mvs else None
        # print("depths_h", batch["near_fars"], depths_h.shape, depths_h[0,0,:,:])
        # volume_feature:(1, 8, D, 176, 208)       img_feat:(B, V, C, h, w)
        cam_expected_depth = None
        ndc_std_depth = None
        # volume_feature: 1, 8, 128, 176, 208;
        # img_feat: 1, 3, 32, 128, 160;
        # depth_values: 1, 128
        photometric_confidence_lst=[]
        cam_xyz_lst = []
        nearfar_mask_lst = []
        volume_prob = None
        # w2c_ref = batch["w2cs"][:, self.args.ref_vid, ...].transpose(1, 2)
        depth_vid = [int(self.args.depth_vid[i]) for i in range(len(self.args.depth_vid))]
        if self.args.manual_depth_view < 1:
            if self.args.manual_depth_view == -1:
                img_feats = self.FeatureNet(imgs[:, :self.args.init_view_num])
            for i in range(len(depth_vid)):
                vid = depth_vid[i]
                if self.args.manual_depth_view == -1:
                    volume_feature, img_feats, depth_values = self.MVSNet(imgs[:, :self.args.init_view_num], img_feats, proj_mats[:, vid, :3], near_fars[0, vid], pad=self.args.pad, vid=vid)
                    volume_prob = self.ProbNet(volume_feature)  # ([1, 1, 128, 176, 208])
                # print("volume_prob", volume_prob.shape)
                elif self.args.manual_depth_view == 0:
                    cam_expected_depth = depths_h[:,vid:vid+1,...]
                    ndc_std_depth = torch.ones_like(cam_expected_depth) * self.args.manual_std_depth
                ndc_xyz, cam_xyz, HDWD, nearfar_mask = self.sample_func(volume_prob, self.args, batch["intrinsics"][:, vid, ...], near_fars[0, vid], cam_expected_depth=cam_expected_depth, ndc_std_depth=ndc_std_depth)

                if cam_xyz.shape[1] > 0:
                    cam_xyz_lst.append(cam_xyz)
                    nearfar_mask_lst.append(nearfar_mask)
        else:
            near_far_depth = batch["near_fars_depth"][0]
            depth_interval, depth_min = (near_far_depth[1] - near_far_depth[0]) / 192., near_far_depth[0]
            depth_values = (depth_min + torch.arange(0, 192, device="cuda", dtype=torch.float32) * depth_interval)[None, :]
            dimgs = batch["mvs_images"] if "mvs_images" in batch else imgs
            bmvs_2d_features=None
            # print("dimgs",dimgs.shape)
            bimgs = dimgs[:, :self.args.init_view_num].expand(len(self.args.depth_vid), -1, -1, -1, -1)
            bvid = torch.as_tensor(depth_vid, dtype=torch.long, device="cuda")
            bproj_mats = proj_mats[0, bvid, ...]
            bdepth_values = depth_values.expand(len(self.args.depth_vid), -1)

            if self.args.manual_depth_view == 1:
                with torch.no_grad():
                    # 1, 128, 160;  1, 128, 160; prob_volume: 1, 192, 128, 160
                    depths_h, photometric_confidence, _, _ = self.MVSNet(bimgs, bproj_mats, bdepth_values, features=bmvs_2d_features)
                    depths_h, photometric_confidence = depths_h[:,None,...], photometric_confidence[:,None,...]
                # B,N,H,W,3,    B,N,H,W,3,      1,      1,1,H,W
            else:
                dnum = self.args.manual_depth_view
                with torch.no_grad():
                    # prob_volume: 1, 192, 128, 160
                    _, prob_sm_volume, prob_raw_volume = self.MVSNet(
                        bimgs, bproj_mats, bdepth_values, features=bmvs_2d_features, prob_only=True)
                    # prob_volume = torch.sigmoid(prob_raw_volume)
                    prob_volume = prob_sm_volume
                    photometric_confidence, topk_idx = torch.topk(prob_volume, dnum, dim=1) # 1, 5, 128, 160; 1, 5, 128, 160

                    depths_h = torch.cat([depth_values[0,topk_idx[i].view(-1)].view(1, dnum, prob_volume.shape[-2], prob_volume.shape[-1]) for i in range(len(depth_vid))], dim=0)

            bcam_expected_depth = torch.nn.functional.interpolate(depths_h, size=list(dimgs.shape)[-2:], mode='nearest')

            photometric_confidence = torch.nn.functional.interpolate(photometric_confidence, size=list(dimgs.shape)[-2:], mode='nearest')  # 1, 1, H, W
            photometric_confidence_lst = torch.unbind(photometric_confidence[:,None,...], dim=0)
            bndc_std_depth = torch.ones_like(bcam_expected_depth) * self.args.manual_std_depth
            for i in range(len(depth_vid)):
                vid = depth_vid[i]
                cam_expected_depth, ndc_std_depth = bcam_expected_depth[i:i+1], bndc_std_depth[i:i+1]
                ndc_xyz, cam_xyz, HDWD, nearfar_mask = self.sample_func(volume_prob, self.args, batch["intrinsics"][:, vid,...], near_fars[0, vid], cam_expected_depth=cam_expected_depth, ndc_std_depth=ndc_std_depth)
                if cam_xyz.shape[1] > 0:
                    cam_xyz_lst.append(cam_xyz)
                    nearfar_mask_lst.append(nearfar_mask)
        return cam_xyz_lst, photometric_confidence_lst, nearfar_mask_lst, HDWD, data_mvs, [batch["intrinsics"][:,int(vid),...] for vid in self.args.depth_vid], [batch["w2cs"][:,int(vid),...] for vid in self.args.depth_vid]



    def forward(self, batch):
        # 3 , 3, 3, 2, 4, dict, 3, 3

        cam_xyz_lst, photometric_confidence_lst, nearfar_mask_lst, HDWD, data_mvs, intrinsics_lst, extrinsics_lst  = self.gen_points(batch)
        # #################### FILTER by Masks ##################
        gpu_filter = True

        if self.args.manual_depth_view != 0:
            # cuda filter
            if gpu_filter:
                cam_xyz_lst, _, photometric_confidence_lst = filter_utils.filter_by_masks_gpu(cam_xyz_lst, intrinsics_lst, extrinsics_lst, photometric_confidence_lst, nearfar_mask_lst, self.args)
            else:
                cam_xyz_lst, _, photometric_confidence_lst = filter_utils.filter_by_masks([cam_xyz.cpu().numpy() for cam_xyz in cam_xyz_lst], [intrinsics.cpu().numpy() for intrinsics in intrinsics_lst], [extrinsics.cpu().numpy() for extrinsics in extrinsics_lst], [confidence.cpu().numpy() for confidence in photometric_confidence_lst], [nearfar_mask.cpu().numpy() for nearfar_mask in nearfar_mask_lst], self.args)
                cam_xyz_lst = [torch.as_tensor(cam_xyz, device="cuda", dtype=torch.float32) for cam_xyz in cam_xyz_lst]
                photometric_confidence_lst = [torch.as_tensor(confidence, device="cuda", dtype=torch.float32) for confidence in photometric_confidence_lst]
        else:
            B, N, C, H, W, _ = cam_xyz_lst[0].shape
            cam_xyz_lst = [cam_xyz.view(C, H, W, 3) for cam_xyz in cam_xyz_lst]
            cam_xyz_lst = [cam_xyz[nearfar_mask[0,...], :] for cam_xyz, nearfar_mask in zip(cam_xyz_lst, nearfar_mask_lst)]
            # print("after filterd", cam_xyz_lst[0].shape)
            photometric_confidence_lst = [torch.ones_like(cam_xyz[...,0]) for cam_xyz in cam_xyz_lst]

        img_feats = self.get_image_features(batch['images'])

        points_features_lst = [self.query_embedding(HDWD, torch.as_tensor(cam_xyz_lst[i][None, ...], device="cuda", dtype=torch.float32), photometric_confidence_lst[i][None, ..., None], img_feats, data_mvs['c2ws'], data_mvs['w2cs'], batch["intrinsics"], int(self.args.depth_vid[i]), pointdir_w=False) for i in range(len(cam_xyz_lst))]


        # #################### start query embedding ##################
        xyz_ref_lst = [(torch.cat([xyz_cam, torch.ones_like(xyz_cam[..., 0:1])], dim=-1) @ torch.linalg.inv(
            cam_extrinsics[0]).transpose(0, 1) @ batch["w2cs"][0, self.args.ref_vid, ...].transpose(0, 1))[..., :3] for
                       xyz_cam, cam_extrinsics in zip(cam_xyz_lst, extrinsics_lst)]
        ref_xyz = torch.cat(xyz_ref_lst, dim=0)
        points_embedding = torch.cat([points_features[0] for points_features in points_features_lst], dim=1)
        points_colors = torch.cat([points_features[1] for points_features in points_features_lst], dim=1) if points_features_lst[0][1] is not None else None
        points_ref_dirs = torch.cat([points_features[2] for points_features in points_features_lst], dim=1) if points_features_lst[0][2] is not None else None
        points_conf = torch.cat([points_features[3] for points_features in points_features_lst], dim=1) if points_features_lst[0][3] is not None else None

        return ref_xyz, points_embedding, points_colors, points_ref_dirs, points_conf


    def save_points(self, xyz, dir, total_steps):
        if xyz.ndim < 3:
            xyz = xyz[None, ...]
        os.makedirs(dir, exist_ok=True)
        for i in range(xyz.shape[0]):
            if isinstance(total_steps, str):
                filename = 'step-{}-{}.txt'.format(total_steps, i)
            else:
                filename = 'step-{:04d}-{}.txt'.format(total_steps, i)
            filepath = os.path.join(dir, filename)
            np.savetxt(filepath, xyz[i, ...].reshape(-1, xyz.shape[-1]), delimiter=";")

    def save_image(self, img_array, filepath):
        assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                             and img_array.shape[2] in [3, 4])

        if img_array.dtype != np.uint8:
            img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        Image.fromarray(img_array).save(filepath)