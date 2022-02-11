from .base_rendering_model import *
from .neural_points.neural_points import NeuralPoints
from .aggregators.point_aggregators import PointAggregator
import os


class NeuralPointsVolumetricModel(BaseRenderingModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        BaseRenderingModel.modify_commandline_options(parser, is_train)
        NeuralPoints.modify_commandline_options(parser, is_train)
        PointAggregator.modify_commandline_options(parser, is_train)

        parser.add_argument(
            '--neural_point_dir',
            type=str,
            default=None,
            help='alternative loading neural_point directory')

        parser.add_argument(
            '--embedding_size',
            type=int,
            default=-1,
            help='number of dimensions for latent code embedding')
        parser.add_argument(
            "--loss_embedding_l2_weight",
            type=float,
            default=-1,
            help="weight for the embedding l2 loss",
        )
        parser.add_argument('--loss_kld_weight',
                            type=float,
                            default=-1,
                            help='weight for the VAE kld')

        # encoder
        parser.add_argument(
            "--compute_depth",
            type=int,
            default=0,
            help=
            "If compute detph or not. If false, depth is only computed when depth is required by losses",
        )


        parser.add_argument(
            "--raydist_mode_unit",
            type=int,
            default=0,
            help="if set raydist max as one voxel",
        )

        parser.add_argument(
            '--save_point_freq',
            type=int,
            default=100000,
            help='frequency of showing training results on console')

        parser.add_argument(
            '--alter_step',
            type=int,
            default=0,
            help='0 for no alter,')

        parser.add_argument(
            '--prob',
            type=int,
            default=0,
            help='will be set as 0 for normal traing and 1 for prob, ')


    def add_default_color_losses(self, opt):
        if "coarse_raycolor" not in opt.color_loss_items:
            opt.color_loss_items.append('coarse_raycolor')
        if opt.fine_sample_num > 0:
            opt.color_loss_items.append('fine_raycolor')

    def add_default_visual_items(self, opt):
        opt.visual_items = ['gt_image', 'coarse_raycolor', 'queried_shading']
        if opt.fine_sample_num > 0:
            opt.visual_items.append('fine_raycolor')

    def run_network_models(self):
        return self.fill_invalid(self.net_ray_marching(**self.input), self.input)

    def fill_invalid(self, output, input):
        # ray_mask:             torch.Size([1, 1024])
        # coarse_is_background: torch.Size([1, 336, 1])  -> 1, 1024, 1
        # coarse_raycolor:      torch.Size([1, 336, 3])  -> 1, 1024, 3
        # coarse_point_opacity: torch.Size([1, 336, 24]) -> 1, 1024, 24
        ray_mask = output["ray_mask"]
        B, OR = ray_mask.shape
        ray_inds = torch.nonzero(ray_mask) # 336, 2
        coarse_is_background_tensor = torch.ones([B, OR, 1], dtype=output["coarse_is_background"].dtype, device=output["coarse_is_background"].device)
        # print("coarse_is_background", output["coarse_is_background"].shape)
        # print("coarse_is_background_tensor", coarse_is_background_tensor.shape)
        # print("ray_inds", ray_inds.shape, ray_mask.shape)
        coarse_is_background_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_is_background"]
        output["coarse_is_background"] = coarse_is_background_tensor
        output['coarse_mask'] = 1 - coarse_is_background_tensor

        if "bg_ray" in self.input:
            coarse_raycolor_tensor = coarse_is_background_tensor * self.input["bg_ray"]
            coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] += output["coarse_raycolor"][0]
        else:
            coarse_raycolor_tensor = self.tonemap_func(
                torch.ones([B, OR, 3], dtype=output["coarse_raycolor"].dtype, device=output["coarse_raycolor"].device) * input["bg_color"][None, ...])
            coarse_raycolor_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_raycolor"]
        output["coarse_raycolor"] = coarse_raycolor_tensor

        coarse_point_opacity_tensor = torch.zeros([B, OR, output["coarse_point_opacity"].shape[2]], dtype=output["coarse_point_opacity"].dtype, device=output["coarse_point_opacity"].device)
        coarse_point_opacity_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["coarse_point_opacity"]
        output["coarse_point_opacity"] = coarse_point_opacity_tensor

        queried_shading_tensor = torch.ones([B, OR, output["queried_shading"].shape[2]], dtype=output["queried_shading"].dtype, device=output["queried_shading"].device)
        queried_shading_tensor[ray_inds[..., 0], ray_inds[..., 1], :] = output["queried_shading"]
        output["queried_shading"] = queried_shading_tensor

        if self.opt.prob == 1 and "ray_max_shading_opacity" in output:
            # print("ray_inds", ray_inds.shape, torch.sum(output["ray_mask"]))
            output = self.unmask(ray_inds, output, ["ray_max_sample_loc_w", "ray_max_shading_opacity", "shading_avg_color", "shading_avg_dir", "shading_avg_conf", "shading_avg_embedding", "ray_max_far_dist"], B, OR)
        return output

    def unmask(self, ray_inds, output, names, B, OR):
        for name in names:
            if output[name] is not None:
                name_tensor = torch.zeros([B, OR, *output[name].shape[2:]], dtype=output[name].dtype, device=output[name].device)
                name_tensor[ray_inds[..., 0], ray_inds[..., 1], ...] = output[name]
                output[name] = name_tensor
        return output

    def get_additional_network_params(self, opt):
        param = {}
        # additional parameters

        self.aggregator = self.check_getAggregator(opt)
        self.is_compute_depth = opt.compute_depth or not not opt.depth_loss_items
        checkpoint_path = os.path.join(opt.checkpoints_dir, opt.name, '{}_net_ray_marching.pth'.format(opt.resume_iter))
        checkpoint_path = checkpoint_path if os.path.isfile(checkpoint_path) else None
        if opt.num_point > 0:
            self.neural_points = NeuralPoints(opt.point_features_dim, opt.num_point, opt, self.device, checkpoint=checkpoint_path, feature_init_method=opt.feature_init_method, reg_weight=0., feedforward=opt.feedforward)
        else:
            self.neural_points = None

        add_property2dict(param, self, [
            'aggregator', 'is_compute_depth', "neural_points", "opt"
        ])
        add_property2dict(param, opt, [
            'num_pos_freqs', 'num_viewdir_freqs'
        ])

        return param

    def create_network_models(self, opt):

        params = self.get_additional_network_params(opt)
        # network
        self.net_ray_marching = NeuralPointsRayMarching(
            **params, **self.found_funcs)

        self.model_names = ['ray_marching'] if getattr(self, "model_names", None) is None else self.model_names + ['ray_marching']

        # parallel
        if self.opt.gpu_ids:
            self.net_ray_marching.to(self.device)
            self.net_ray_marching = torch.nn.DataParallel(
                self.net_ray_marching, self.opt.gpu_ids)


    def check_getAggregator(self, opt, **kwargs):
        aggregator = PointAggregator(opt)
        return aggregator


    def setup_optimizer(self, opt):
        '''
            Setup the optimizers for all networks.
            This assumes network modules have been added to self.model_names
            By default, it uses an adam optimizer for all parameters.
        '''

        net_params = []
        neural_params = []
        for name in self.model_names:
            net = getattr(self, 'net_' + name)
            param_lst = list(net.named_parameters())

            net_params = net_params + [par[1] for par in param_lst if not par[0].startswith("module.neural_points")]
            neural_params = neural_params + [par[1] for par in param_lst if par[0].startswith("module.neural_points")]

        self.net_params = net_params
        self.neural_params = neural_params

        # opt.lr=0
        self.optimizer = torch.optim.Adam(net_params,
                                          lr=opt.lr,
                                          betas=(0.9, 0.999))
        self.neural_point_optimizer = torch.optim.Adam(neural_params,
                                          lr=opt.lr, #/ 5.0,
                                          betas=(0.9, 0.999))
        self.optimizers = [self.optimizer, self.neural_point_optimizer]

    def backward(self, iters):
        [optimizer.zero_grad() for optimizer in self.optimizers]
        if self.opt.is_train:
            self.loss_total.backward()
            if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 0:
                self.optimizer.step()
            if self.opt.alter_step == 0 or int(iters / self.opt.alter_step) % 2 == 1:
                self.neural_point_optimizer.step()


    def optimize_parameters(self, backward=True, total_steps=0):
        self.forward()
        self.update_rank_ray_miss(total_steps)
        self.backward(total_steps)

    def update_rank_ray_miss(self, total_steps):
        raise NotImplementedError

class NeuralPointsRayMarching(nn.Module):
    def __init__(self,
             tonemap_func=None,
             render_func=None,
             blend_func=None,
             aggregator=None,
             is_compute_depth=False,
             neural_points=None,
             opt=None,
             num_pos_freqs=0,
             num_viewdir_freqs=0,
             **kwargs):
        super(NeuralPointsRayMarching, self).__init__()

        self.aggregator = aggregator

        self.num_pos_freqs = num_pos_freqs
        self.num_viewdir_freqs = num_viewdir_freqs
        # ray generation

        self.render_func = render_func
        self.blend_func = blend_func

        self.tone_map = tonemap_func
        self.return_depth = is_compute_depth
        self.return_color = True
        self.opt = opt
        self.neural_points = neural_points


    def forward(self,
                campos,
                raydir,
                gt_image=None,
                bg_color=None,
                camrotc2w=None,
                pixel_idx=None,
                near=None,
                far=None,
                focal=None,
                h=None,
                w=None,
                intrinsic=None,
                **kargs):
        output = {}
        # B, channel, 292, 24, 32;      B, 3, 294, 24, 32;     B, 294, 24;     B, 291, 2
        sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, ray_mask_tensor, vsize, grid_vox_sz = self.neural_points({"pixel_idx": pixel_idx, "camrotc2w": camrotc2w, "campos": campos, "near": near, "far": far,"focal": focal, "h": h, "w": w, "intrinsic": intrinsic,"gt_image":gt_image, "raydir":raydir})

        decoded_features, ray_valid, weight, conf_coefficient = self.aggregator(sampled_color, sampled_Rw2c, sampled_dir, sampled_conf, sampled_embedding, sampled_xyz_pers, sampled_xyz, sample_pnt_mask, sample_loc, sample_loc_w, sample_ray_dirs, vsize, grid_vox_sz)
        ray_dist = torch.cummax(sample_loc[..., 2], dim=-1)[0]
        ray_dist = torch.cat([ray_dist[..., 1:] - ray_dist[..., :-1], torch.full((ray_dist.shape[0], ray_dist.shape[1], 1), vsize[2], device=ray_dist.device)], dim=-1)

        mask = ray_dist < 1e-8
        if self.opt.raydist_mode_unit > 0:
            mask = torch.logical_or(mask, ray_dist > 2 * vsize[2])
        mask = mask.to(torch.float32)
        ray_dist = ray_dist * (1.0 - mask) + mask * vsize[2]
        ray_dist *= ray_valid.float()
        # raydir: N x Rays x 3sampled_color
        # raypos: N x Rays x Samples x 3
        # ray_dist: N x Rays x Samples
        # ray_valid: N x Rays x Samples
        # ray_features: N x Rays x Samples x Features
        # Output
        # ray_color: N x Rays x 3
        # point_color: N x Rays x Samples x 3
        # opacity: N x Rays x Samples
        # acc_transmission: N x Rays x Samples
        # blend_weight: N x Rays x Samples x 1
        # background_transmission: N x Rays x 1
        # ray march
        output["queried_shading"] = torch.logical_not(torch.any(ray_valid, dim=-1, keepdims=True)).repeat(1, 1, 3).to(torch.float32)
        if self.return_color:
            if "bg_ray" in kargs:
                bg_color = None
            (
                ray_color,
                point_color,
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = ray_march(ray_dist, ray_valid, decoded_features, self.render_func, self.blend_func, bg_color)
            ray_color = self.tone_map(ray_color)
            output["coarse_raycolor"] = ray_color
            output["coarse_point_opacity"] = opacity
        else:
            (
                opacity,
                acc_transmission,
                blend_weight,
                background_transmission,
                _,
            ) = alpha_ray_march(ray_dist, ray_valid, decoded_features, self.blend_func)

        if self.return_depth:
            alpha_blend_weight = opacity * acc_transmission
            weight = alpha_blend_weight.view(alpha_blend_weight.shape[:3])
            avg_depth = (weight * ray_ts).sum(-1) / (weight.sum(-1) + 1e-6)
            output["coarse_depth"] = avg_depth
        output["coarse_is_background"] = background_transmission
        output["ray_mask"] = ray_mask_tensor
        if weight is not None:
            output["weight"] = weight.detach()
            output["blend_weight"] = blend_weight.detach()
            output["conf_coefficient"] = conf_coefficient


        if self.opt.prob == 1 and output["coarse_point_opacity"].shape[1] > 0 :
            B, OR, _, _ = sample_pnt_mask.shape
            if weight is not None:
                output["ray_max_shading_opacity"], opacity_ind = torch.max(output["coarse_point_opacity"], dim=-1, keepdim=True)
                opacity_ind=opacity_ind[..., None] # 1, 1024, 1, 1

                output["ray_max_sample_loc_w"] = torch.gather(sample_loc_w, 2, opacity_ind.expand(-1, -1, -1, sample_loc_w.shape[-1])).squeeze(2) # 1, 1024, 24, 3 -> 1, 1024, 3
                weight = torch.gather(weight*conf_coefficient, 2, opacity_ind.expand(-1, -1, -1, weight.shape[-1])).squeeze(2)[..., None] # 1, 1024, 8
                opacity_ind = opacity_ind[...,None]

                sampled_xyz_max_opacity = torch.gather(sampled_xyz, 2, opacity_ind.expand(-1, -1, -1, sampled_xyz.shape[-2], sampled_xyz.shape[-1])).squeeze(2) # 1, 1024, 8, 3
                output["ray_max_far_dist"] = torch.min(torch.norm(sampled_xyz_max_opacity - output["ray_max_sample_loc_w"][..., None,:], dim=-1), axis=-1, keepdim=True)[0]

                sampled_color = torch.gather(sampled_color, 2, opacity_ind.expand(-1, -1, -1, sampled_color.shape[-2], sampled_color.shape[-1])).squeeze(2) if sampled_color is not None else None # 1, 1024, 8, 3
                sampled_dir = torch.gather(sampled_dir, 2, opacity_ind.expand(-1, -1, -1, sampled_dir.shape[-2], sampled_dir.shape[-1])).squeeze(2)  if sampled_dir is not None else None # 1, 1024, 8, 3
                sampled_conf = torch.gather(sampled_conf, 2, opacity_ind.expand(-1, -1, -1, sampled_conf.shape[-2], sampled_conf.shape[-1])).squeeze(2)  if sampled_conf is not None else None # 1, 1024, 8, 1
                sampled_embedding = torch.gather(sampled_embedding, 2, opacity_ind.expand(-1, -1, -1, sampled_embedding.shape[-2], sampled_embedding.shape[-1])).squeeze(2) # 1, 1024, 8, 1

                output["shading_avg_color"] = torch.sum(sampled_color * weight, dim=-2)  if sampled_color is not None else None
                output["shading_avg_dir"] = torch.sum(sampled_dir * weight, dim=-2) if sampled_dir is not None else None
                output["shading_avg_conf"] = torch.sum(sampled_conf * weight, dim=-2) if sampled_conf is not None else None
                output["shading_avg_embedding"] = torch.sum(sampled_embedding * weight, dim=-2)
            else:
                output.update({
                    "ray_max_shading_opacity": torch.zeros([0, 0, 1, 1], device="cuda"),
                    "ray_max_sample_loc_w": torch.zeros([0, 0, 3], device="cuda"),
                    "ray_max_far_dist": torch.zeros([0, 0, 1], device="cuda"),
                    "shading_avg_color": torch.zeros([0, 0, 3], device="cuda"),
                    "shading_avg_dir": torch.zeros([0, 0, 3], device="cuda"),
                    "shading_avg_conf": torch.zeros([0, 0, 1], device="cuda"),
                    "shading_avg_embedding": torch.zeros([0, 0, sampled_embedding.shape[-1]], device="cuda"),
                })

        return output