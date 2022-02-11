from .base_rendering_model import *
from .neural_points_volumetric_model import NeuralPointsVolumetricModel
from .neural_points.neural_points import NeuralPoints
from .mvs.mvs_points_model import MvsPointsModel
from .mvs import mvs_utils
from. import base_model
from .aggregators.point_aggregators import PointAggregator
import os
import torch.nn.functional as F
import time
from utils import format as fmt


class MvsPointsVolumetricModel(NeuralPointsVolumetricModel):

    def __init__(self,):
        super().__init__()
        self.optimizer, self.neural_point_optimizer, self.output, self.raygen_func, self.render_func, self.blend_func, self.coarse_raycolor, self.gt_image, self.input, self.l1loss, self.l2loss, self.tonemap_func, self.top_ray_miss_ids, self.top_ray_miss_loss, self.loss_ray_masked_coarse_raycolor, self.loss_ray_miss_coarse_raycolor, self.loss_total, self.loss_coarse_raycolor, self.loss_conf_coefficient = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    #
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        MvsPointsModel.modify_commandline_options(parser, is_train)
        NeuralPointsVolumetricModel.modify_commandline_options(parser, is_train=is_train)
        parser.add_argument(
            '--mode',
            type=int,
            default=0,
            help='0 for both mvs and pointnerf, 1 for only mvs, 2 for only pointnerf')
        parser.add_argument(
            '--add_shading_dist',
            type=int,
            default=0,
            help='0 for both mvs and pointnerf, 1 for only mvs, 2 for only pointnerf')



    def create_network_models(self, opt):
        if opt.mode != 2:
            self.net_mvs = MvsPointsModel(opt).to(self.device)
            self.model_names = ['mvs']

        if opt.mode != 1:
            super(MvsPointsVolumetricModel, self).create_network_models(opt)


    def setup_optimizer(self, opt):
        '''
            Setup the optimizers for all networks.
            This assumes network modules have been added to self.model_names
            By default, it uses an adam optimizer for all parameters.
        '''

        net_params = []
        neural_params = []
        mvs_params = []
        self.optimizers = []
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            if name == "mvs":
                # print([[par[0], torch.typename(par[1])] for par in net.named_parameters()])
                param_lst = list(net.named_parameters())
                mvs_params = mvs_params + [par[1] for par in param_lst]
            else:
                param_lst = list(net.named_parameters())

                net_params = net_params + [par[1] for par in param_lst if not par[0].startswith("module.neural_points")]
                neural_params = neural_params + [par[1] for par in param_lst if par[0].startswith("module.neural_points")]

        self.net_params = net_params
        self.neural_params = neural_params
        self.mvs_params = mvs_params
        mvs_lr = opt.mvs_lr if opt.mvs_lr is not None else opt.lr

        if len(mvs_params) > 0:
            self.mvs_optimizer = torch.optim.Adam(mvs_params,
                                              lr=mvs_lr,
                                              betas=(0.9, 0.999))
            self.optimizers.append(self.mvs_optimizer)

        if len(net_params) > 0:
            self.optimizer = torch.optim.Adam(net_params,
                                          lr=opt.lr,
                                          betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)

        if len(neural_params) > 0:
            self.neural_point_optimizer = torch.optim.Adam(neural_params,
                                          lr=opt.plr, #/ 5.0,
                                          betas=(0.9, 0.999))
            self.optimizers.append(self.neural_point_optimizer)
            print("neural_params", [(par[0], par[1].shape, par[1].requires_grad)  for par in param_lst if par[0].startswith("module.neural_points")])
        else:
            # When not doing per-scene optimization
            print("no neural points as nn.Parameter")


    def backward(self, iters):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        if self.opt.is_train:
            # print("self.loss_total", self.ray_masked_coarse_color.grad)
            # print("self.loss_total", self.loss_total)
            if self.loss_total != 0:
                self.loss_total.backward()
            else:
                print(fmt.RED + "Loss == 0" +
                      fmt.END)

            if self.opt.feedforward:
                if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 0:
                    self.optimizer.step()
                if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 1:
                    self.mvs_optimizer.step()
            else:
                if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 0:
                    self.optimizer.step()
                if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 1:
                    self.neural_point_optimizer.step()


    def forward(self):
        if self.opt.mode != 2:
            points_xyz, points_embedding, points_colors, points_dirs, points_conf = self.net_mvs(self.input)
            # print("volume_feature", volume_feature.shape)
            self.neural_points.set_points(points_xyz, points_embedding, points_color=points_colors, points_dir=points_dirs, points_conf=points_conf, parameter=self.opt.feedforward==0) # if feedforward, no neural points optimization
        self.output = self.run_network_models()
        if "depths_h" in self.input:
            depth_gt = self.input["depths_h"][:,self.opt.trgt_id,...] if self.input["depths_h"].dim() > 3 else self.input["depths_h"]
            self.output["ray_depth_mask"] = depth_gt > 0
        self.set_visuals()
        if not self.opt.no_loss:
            self.compute_losses()

    def update_rank_ray_miss(self, total_steps):
        if (self.opt.prob_kernel_size is None or np.sum(np.asarray(self.opt.prob_tiers) < total_steps) < (len(self.opt.prob_kernel_size) // 3)):

            if self.opt.prob_freq > 0 and self.opt.prob_num_step > 1:
                self.top_ray_miss_loss, self.top_ray_miss_ids = self.rank_ray_miss(self.input["id"][0], self.loss_ray_miss_coarse_raycolor, self.top_ray_miss_ids, self.top_ray_miss_loss)
            elif self.opt.prob_freq > 0 and self.opt.prob_num_step == 1:
                self.top_ray_miss_loss[0] = max(self.loss_ray_miss_coarse_raycolor, self.top_ray_miss_loss[0])


    def rank_ray_miss(self, new_id, newloss, inds, losses):
        with torch.no_grad():
            mask = (inds - new_id) == 0
            if torch.sum(mask) > 0:
                losses[mask] = max(newloss, losses[mask])
            else:
                inds[-1] = new_id
                losses[-1] = newloss
            losses, indices = torch.sort(losses, descending=True)
            inds = inds[indices]
            return losses, inds

    def setup(self, opt, train_len=None):
        super(MvsPointsVolumetricModel, self).setup(opt)
        if opt.prob_freq > 0 and train_len is not None and opt.prob_num_step > 1:
            self.num_probe = train_len // opt.prob_num_step
            self.reset_ray_miss_ranking()
        elif opt.prob_freq > 0 and train_len is not None and opt.prob_num_step == 1:
            self.top_ray_miss_loss=torch.zeros([1], dtype=torch.float32, device=self.device)


    def reset_ray_miss_ranking(self):
        self.top_ray_miss_loss = torch.zeros([self.num_probe + 1], dtype=torch.float32, device=self.device)
        self.top_ray_miss_ids = torch.arange(self.num_probe + 1, dtype=torch.int32, device=self.device)

    def set_points(self, points_xyz, points_embedding, points_color=None, points_dir=None, points_conf=None, Rw2c=None, eulers=None, editing=False):
        if not editing:
            self.neural_points.set_points(points_xyz, points_embedding, points_color=points_color, points_dir=points_dir, points_conf=points_conf, parameter=self.opt.feedforward == 0, Rw2c=Rw2c, eulers=eulers)
        else:
            self.neural_points.editing_set_points(points_xyz, points_embedding, points_color=points_color, points_dir=points_dir, points_conf=points_conf, parameter=self.opt.feedforward == 0, Rw2c=Rw2c, eulers=eulers)
        if self.opt.feedforward == 0 and self.opt.is_train:
            self.setup_optimizer(self.opt)


    def prune_points(self, thresh):
        self.neural_points.prune(thresh)

    def clean_optimizer_scheduler(self):
        # self.neural_points.querier.clean_up()
        self.optimizers.clear()
        self.schedulers.clear()
        self.neural_params.clear()
        self.mvs_params.clear()
        # self.optimizer.cpu(), self.neural_point_optimizer.cpu()
        del self.optimizer, self.neural_point_optimizer, self.optimizers, self.schedulers, self.mvs_params, self.neural_params


    def reset_optimizer(self, opt):
        self.clean_optimizer(opt)
        self.setup_optimizer(opt)

    def clean_optimizer(self):
        self.optimizers.clear()
        self.net_params.clear()
        self.neural_params.clear()
        self.mvs_params.clear()
        del self.optimizer, self.neural_point_optimizer, self.net_params, self.neural_params, self.mvs_params

    def clean_scheduler(self):
        for scheduler in self.schedulers:
            del scheduler
        self.schedulers.clear()
        del self.schedulers

    def init_scheduler(self, total_steps, opt):
        self.schedulers = [
            base_model.get_scheduler(optim, opt) for optim in self.optimizers
        ]
        if total_steps > 0:
            for scheduler in self.schedulers:
                for i in range(total_steps):
                    scheduler.step()

    def reset_scheduler(self, total_steps, opt):
        self.schedulers.clear()
        self.schedulers = [
            base_model.get_scheduler(optim, opt) for optim in self.optimizers
        ]
        if total_steps > 0:
            for scheduler in self.schedulers:
                for i in range(total_steps):
                    scheduler.step()

    def gen_points(self):
        cam_xyz_lst, photometric_confidence_lst, point_mask_lst, HDWD, data_mvs, intrinsics_lst, extrinsics_lst = self.net_mvs.gen_points(self.input)
        # print("cam_xyz_lst", cam_xyz_lst[0].shape, torch.min(cam_xyz_lst[0].view(-1,3), dim=-2)[0], torch.max(cam_xyz_lst[0].view(-1,3), dim=-2)[0])
        # self.net_mvs.gen_bg_points(self.input)
        return cam_xyz_lst, photometric_confidence_lst, point_mask_lst, intrinsics_lst, extrinsics_lst, HDWD, data_mvs['c2ws'], data_mvs['w2cs'], self.input["intrinsics"], self.input["near_fars"]


    def query_embedding(self, HDWD, cam_xyz, photometric_confidence, imgs, c2ws, w2cs, intrinsics, cam_vid, pointdir_w=True):
        img_feats = self.net_mvs.get_image_features(imgs)
        return self.net_mvs.query_embedding(HDWD, cam_xyz, photometric_confidence, img_feats, c2ws, w2cs, intrinsics, cam_vid, pointdir_w=pointdir_w)


    def grow_points(self, points_xyz, points_embedding, points_color, points_dir, points_conf):
        self.neural_points.grow_points(points_xyz, points_embedding, points_color, points_dir, points_conf)
        # self.neural_points.reset_querier()

    def cleanup(self):
        if hasattr(self, "neural_points"):
            self.neural_points.querier.clean_up()
            del self.neural_points.querier
            self.neural_points.cpu()
            del self.neural_points
        print("self.model_names", self.model_names)
        if hasattr(self, "net_ray_marching"):
            self.net_ray_marching.cpu()
            del self.net_ray_marching
        if hasattr(self, "net_mvs"):
            self.net_mvs.cpu()
            del self.net_mvs
        if hasattr(self, "net_params"):
            self.net_params.clear()
            del self.net_params
        if hasattr(self, "neural_params"):
            self.neural_params.clear()
            del self.neural_params
        if hasattr(self, "mvs_params"):
            self.mvs_params.clear()
            del self.mvs_params
        if hasattr(self, "aggregator"):
            self.aggregator.cpu()
            del self.aggregator
        if hasattr(self, "optimizers"):
            self.optimizers.clear()
            self.schedulers.clear()
        del self.optimizer,  self.neural_point_optimizer, self.output, self.raygen_func, self.render_func, self.blend_func, self.coarse_raycolor, self.gt_image, self.input, self.l1loss, self.l2loss, self.tonemap_func, self.top_ray_miss_ids, self.top_ray_miss_loss, self.loss_ray_masked_coarse_raycolor, self.loss_ray_miss_coarse_raycolor, self.loss_total, self.loss_coarse_raycolor, self.loss_conf_coefficient

    def set_bg(self, xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, plane_color, fg_masks=None,**kwargs):
        warped_feats = []
        c2w = torch.eye(4, device="cuda", dtype=torch.float32)[None, ...]  # c2w[:,0,...].cuda()
        count=0
        mask_lst = []
        fg_mask_lst = []
        for imgs, w2c, intrinsics, HDWD in zip(img_lst, w2cs_lst, intrinsics_all, HDWD_lst):
            # "fg_2d_masks", [1, 1, 512, 640]
            # c2w: 1, 3, 4, 4,      w2c: 1, 3, 4, 4,        intrinsics: 1, 3, 3
            HD, WD = HDWD[0], HDWD[1]
            w2c = w2c[:,0,...]
            warp = mvs_utils.homo_warp_nongrid # homo_warp_nongrid_occ if self.args.depth_occ > 0 else homo_warp_nongrid
            src_grid, mask, hard_id_xy = warp(c2w, w2c, intrinsics, xyz_world_sect_plane, HD, WD, filter=False, tolerate=0.1)
            hard_id_xy_valid = hard_id_xy[:, mask[0,:,0], :]

            if fg_masks is None:
                fg_mask = mvs_utils.homo_warp_fg_mask(c2w, w2c, intrinsics, self.neural_points.xyz[None,...], HD, WD, tolerate=0.1)
                fg_mask_lst.append(fg_mask)
            else:
                fg_mask = fg_masks[:,count,...]

            mask[0,mask[0,...,0].clone(),0] = (fg_mask[hard_id_xy_valid[0, ..., 1].long(), hard_id_xy_valid[
                0, ..., 0].long()] < 1)
            # src_grid: 1, 2032, 2
            src_grid = src_grid[:, mask[0,...,0], :]
            mask_lst.append(mask[0,...,0])
            warped_src_feat = mvs_utils.extract_from_2d_grid(imgs[0:1, ...], src_grid.cpu(), mask.cpu())
            warped_feats.append(warped_src_feat.cuda())
            count+=1
        # masks = ~torch.stack(mask_lst, dim = -1) # 2304, 16 fg

        warped_feats = torch.stack(warped_feats, dim=-2) # 1, 2304, 16, 3
        thresh=0.03
        fit_mask = torch.prod(torch.logical_and(warped_feats >= (plane_color - thresh), (warped_feats <= plane_color + thresh)), dim=-1)
        nofit_feats_inds = (1-fit_mask).nonzero()  # 1, 2304, 16
        warped_feats[0, nofit_feats_inds[...,1], nofit_feats_inds[...,2], :] = 0
        warped_feats = torch.max(warped_feats, dim=-2)[0]
        fg_masks = torch.stack(fg_mask_lst, dim=1) if fg_mask_lst is None else fg_masks
        return warped_feats, fg_masks #


    def load_networks(self, epoch):
        for name, net in zip(self.model_names, self.get_networks()):
            assert isinstance(name, str)
            load_filename = '{}_net_{}.pth'.format(epoch, name)
            load_path = os.path.join(self.opt.resume_dir, load_filename)
            print('loading', name, " from ", load_path)
            if not os.path.isfile(load_path):
                print('cannot load', load_path)
                continue
            state_dict = torch.load(load_path, map_location=self.device)
            if epoch=="best" and name == "ray_marching" and self.opt.default_conf > 0.0 and self.opt.default_conf <= 1.0 and self.neural_points.points_conf is not None:
                assert "neural_points.points_conf" not in state_dict
                state_dict["neural_points.points_conf"] = torch.ones_like(self.net_ray_marching.module.neural_points.points_conf) * self.opt.default_conf
            if isinstance(net, nn.DataParallel):
                net = net.module
            net.load_state_dict(state_dict, strict=False)

    def test(self, gen_points=False):
        with torch.no_grad():
            if gen_points:
                self.forward()
            else:
                self.output = self.run_network_models()
                if "depths_h" in self.input:
                    depth_gt = self.input["depths_h"][:, self.opt.trgt_id, ...] if self.input["depths_h"].dim() > 3 else self.input["depths_h"]

                    self.output["ray_depth_mask"] = depth_gt > 0

                self.set_visuals()
                if not self.opt.no_loss:
                    self.compute_losses()
            return self.output
