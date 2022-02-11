import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def find_ray_generation_method(name):
    assert isinstance(name, str), 'ray generation method name must be string'
    if name == 'cube':
        return cube_ray_generation
    elif name == 'near_far_linear':
        return near_far_linear_ray_generation
    elif name == 'near_far_disparity_linear':
        return near_far_disparity_linear_ray_generation
    elif name == 'nerf_near_far_disparity_linear':
        return nerf_near_far_disparity_linear_ray_generation
    elif name == 'nerf_near_far_linear':
        return nerf_near_far_linear_ray_generation
    elif name == 'near_middle_far':
        return near_middle_far_ray_generation
    raise RuntimeError('No such ray generation method: ' + name)


def find_refined_ray_generation_method(name):
    assert isinstance(name, str), 'ray generation method name must be string'
    if name == 'cube':
        return refine_cube_ray_generation
    elif name.startswith('nerf'):
        return nerf_refine_ray_generation
    else:
        #hack default
        return refine_ray_generation
    raise RuntimeError('No such refined ray generation method: ' + name)


def sample_pdf(in_bins, in_weights, n_samples, det=False):
    # bins: N x R x S x 1
    # weights: N x R x s x 1
    in_shape = in_bins.shape
    device = in_weights.device

    bins = in_bins.data.cpu().numpy().reshape([-1, in_shape[2]])
    bins = 0.5 * (bins[..., 1:] + bins[..., :-1])
    # bins: [NR x (S-1)]

    weights = in_weights.data.cpu().numpy().reshape([-1, in_shape[2]])
    weights = weights[..., 1:-1]
    # weights: [NR x (S-2)]

    weights += 1e-5
    pdf = weights / np.sum(weights, axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, axis=-1)
    cdf = np.concatenate([np.zeros_like(cdf[..., :1]), cdf], -1)
    # cdf: [NR x (S-1)]

    if det:
        ur = np.broadcast_to(np.linspace(0, 1, n_samples, dtype=np.float32),
                             (cdf.shape[0], n_samples))
    else:
        ur = np.random.rand(cdf.shape[0], n_samples).astype(np.float32)
    # u: [NR x S2]

    inds = np.stack(
        [np.searchsorted(a, i, side='right') for a, i in zip(cdf, ur)])
    below = np.maximum(0, inds - 1)
    above = np.minimum(cdf.shape[-1] - 1, inds)
    cdf_below = np.take_along_axis(cdf, below, 1)
    cdf_above = np.take_along_axis(cdf, above, 1)
    bins_below = np.take_along_axis(bins, below, 1)
    bins_above = np.take_along_axis(bins, above, 1)
    denom = cdf_above - cdf_below
    denom = np.where(denom < 1e-5, np.ones_like(denom), denom)
    t = (ur - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    samples = torch.from_numpy(samples).view(
        (in_shape[0], in_shape[1], n_samples, 1)).to(device)
    samples = torch.cat([samples, in_bins.detach()], dim=-2)
    samples, _ = torch.sort(samples, dim=-2)
    samples = samples.detach()

    return samples


# def sample_pdf(in_bins, in_weights, n_samples, det=False):
#     # bins: N x R x S x 1
#     # weights: N x R x S x 1
#     import tensorflow as tf
#     tf.config.set_visible_devices([], 'GPU')

#     ori_shape = in_bins.shape
#     device = in_weights.device

#     # bins: (N*R, S)
#     bins = tf.convert_to_tensor(in_bins.data.cpu().numpy().reshape((-1, in_bins.shape[-2])))
#     weights = tf.convert_to_tensor(in_weights.data.cpu().numpy().reshape((-1, in_weights.shape[-2])))

#     bins = 0.5 * (bins[..., 1:] + bins[..., :-1])
#     weights = weights[..., 1:-1]

#     # Get pdf
#     weights += 1e-5  # prevent nans
#     pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
#     cdf = tf.cumsum(pdf, -1)
#     cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

#     # Take uniform samples
#     if det:
#         u = tf.linspace(0., 1., n_samples)
#         u = tf.broadcast_to(u, list(cdf.shape[:-1]) + [n_samples])
#     else:
#         u = tf.random.uniform(list(cdf.shape[:-1]) + [n_samples])

#     # Invert CDF
#     inds = tf.searchsorted(cdf, u, side='right')
#     below = tf.maximum(0, inds - 1)
#     above = tf.minimum(cdf.shape[-1] - 1, inds)
#     inds_g = tf.stack([below, above], -1)
#     cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)
#     bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape) - 2)

#     denom = (cdf_g[..., 1] - cdf_g[..., 0])
#     denom = tf.where(denom < 1e-5, tf.ones_like(denom), denom)
#     t = (u - cdf_g[..., 0]) / denom
#     samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

#     # N x R x N_samples x 1
#     samples = torch.from_numpy(samples.numpy()).view(
#         (in_bins.shape[0], in_bins.shape[1], n_samples, 1)).to(in_bins.device)

#     #  print(samples[0,0,:, 0])
#     #  print(in_bins[0,0,:, 0])
#     # N x R x (N_samples + S) x 1
#     samples = torch.cat([samples, in_bins.detach()], dim=-2)
#     #  samples = torch.cat([samples, in_bins.data], dim=-2)
#     samples, _ = torch.sort(samples, dim=-2)
#     samples = samples.detach()

#     return samples


def near_middle_far_ray_generation(campos,
                                   raydir,
                                   point_count,
                                   near=0.1,
                                   middle=2,
                                   far=10,
                                   middle_split=0.6,
                                   jitter=0.,
                                   **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples

    tvals = torch.linspace(0,
                           1,
                           int(point_count * middle_split) + 1,
                           device=campos.device).view(1, -1)
    vals0 = near * (1 - tvals) + middle * tvals  # N x 1 x Sammples
    tvals = torch.linspace(0,
                           1,
                           int(point_count * (1 - middle_split)) + 2,
                           device=campos.device).view(1, -1)
    vals1 = 1. / (1. / middle *
                  (1 - tvals) + 1. / far * tvals)  # N x 1 x Sammples
    tvals = torch.cat([vals0, vals1], 2)

    segment_length = (tvals[..., 1:] - tvals[..., :-1]) * (
        1 + jitter * (torch.rand(
            (raydir.shape[0], raydir.shape[1], tvals.shape[-1] - 1),
            device=campos.device) - 0.5))
    segment_length = segment_length[..., :point_count]

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat([
        torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device), end_point_ts
    ],
                             dim=2)
    end_point_ts = near + end_point_ts

    middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
    raypos = campos[:, None,
                    None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :,
                                                                       None]
    valid = torch.ones_like(middle_point_ts,
                            dtype=middle_point_ts.dtype,
                            device=middle_point_ts.device)

    return raypos, segment_length, valid, middle_point_ts


def near_far_disparity_linear_ray_generation(campos,
                                             raydir,
                                             point_count,
                                             near=0.1,
                                             far=10,
                                             jitter=0.,
                                             **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples

    tvals = torch.linspace(0, 1, point_count + 1,
                           device=campos.device).view(1, -1)
    tvals = 1. / (1. / near *
                  (1 - tvals) + 1. / far * tvals)  # N x 1 x Sammples
    segment_length = (tvals[..., 1:] -
                      tvals[..., :-1]) * (1 + jitter * (torch.rand(
                          (raydir.shape[0], raydir.shape[1], point_count),
                          device=campos.device) - 0.5))

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat([
        torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device), end_point_ts
    ], dim=2)
    end_point_ts = near + end_point_ts

    middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
    raypos = campos[:, None,
                    None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :,
                                                                       None]
    # print(tvals.shape, segment_length.shape, end_point_ts.shape, middle_point_ts.shape, raypos.shape)
    valid = torch.ones_like(middle_point_ts,
                            dtype=middle_point_ts.dtype,
                            device=middle_point_ts.device)
    # print("campos", campos.shape, campos[0])
    # print("raydir", raydir.shape, raydir[0,0])
    # print("middle_point_ts", middle_point_ts.shape, middle_point_ts[0,0])
    # print("raypos", raypos.shape, raypos[0,0])

    return raypos, segment_length, valid, middle_point_ts


def nerf_near_far_disparity_linear_ray_generation(campos,
                                             raydir,
                                             point_count,
                                             near=0.1,
                                             far=10,
                                             jitter=1.,
                                             **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples


    tvals = torch.linspace(0, 1, point_count,
                           device=campos.device).view(1, -1)
    tvals = 1. / (1. / near *
                  (1 - tvals) + 1. / far * tvals)  # N x 1 x Sammples
    if jitter > 0.0:
        mids = .5 * (tvals[..., 1:] + tvals[..., :-1])
        upper = torch.cat([mids, tvals[..., -1:]], -1)
        lower = torch.cat([tvals[..., :1], mids], -1)
        t_rand = torch.rand([tvals.shape[0],raydir.shape[1],tvals.shape[2]], device=campos.device)
        tvals = lower + (upper - lower) * t_rand
        # print("tvals, {}, t_rand {}, mids {}, upper {}, lower {}".format(tvals.shape, t_rand.shape, mids.shape, upper.shape, lower.shape))
    segment_length = torch.cat([tvals[..., 1:] - tvals[..., :-1], torch.full((tvals.shape[0], tvals.shape[1], 1), 1e10, device=tvals.device)], axis=-1) * torch.linalg.norm(raydir[..., None, :], axis=-1)
    # print("segment_length, {}".format(segment_length.shape))

    raypos = campos[:, None,
                    None, :] + raydir[:, :, None, :] * tvals[:, :, :, None]
    # print("raypos, {}, campos {}, raydir {}, tvals {}".format(raypos.shape, campos.shape, raydir.shape, tvals.shape))

    # print("raypos", raypos[0])
    valid = torch.ones_like(tvals,
                            dtype=raypos.dtype,
                            device=raypos.device)
    # print("campos", campos.shape, campos[0])
    # print("raydir", raydir.shape, raydir[0,0])
    # print("middle_point_ts", middle_point_ts.shape, middle_point_ts[0,0])
    # print("raypos", raypos.shape, raypos[0,0])

    return raypos, segment_length, valid, tvals


def nerf_near_far_linear_ray_generation(campos,
                                             raydir,
                                             point_count,
                                             near=0.1,
                                             far=10,
                                             jitter=1.,
                                             **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples


    tvals = torch.linspace(0, 1, point_count,
                           device=campos.device).view(1, -1)
    tvals = near * (1.-tvals) + far * (tvals)  # N x 1 x Sammples
    if jitter > 0.0:
        mids = .5 * (tvals[..., 1:] + tvals[..., :-1])
        upper = torch.cat([mids, tvals[..., -1:]], -1)
        lower = torch.cat([tvals[..., :1], mids], -1)
        t_rand = torch.rand([tvals.shape[0],raydir.shape[1],tvals.shape[2]], device=campos.device)
        tvals = lower + (upper - lower) * t_rand
        # print("tvals, {}, t_rand {}, mids {}, upper {}, lower {}".format(tvals.shape, t_rand.shape, mids.shape, upper.shape, lower.shape))
    segment_length = torch.cat([tvals[..., 1:] - tvals[..., :-1], torch.full((tvals.shape[0], tvals.shape[1], 1), 1e10, device=tvals.device)], axis=-1) * torch.linalg.norm(raydir[..., None, :], axis=-1)
    raypos = campos[:, None, None, :] + raydir[:, :, None, :] * tvals[:, :, :, None]
    # print("raypos, {}, campos {}, raydir {}, tvals {}".format(raypos.shape, campos.shape, raydir.shape, tvals.shape))

    # print("raypos", raypos[0])
    valid = torch.ones_like(tvals,
                            dtype=raypos.dtype,
                            device=raypos.device)
    # print("campos", campos.shape, campos[0])
    # print("raydir", raydir.shape, raydir[0,0])
    # print("middle_point_ts", middle_point_ts.shape, middle_point_ts[0,0])
    # print("raypos", raypos.shape, raypos[0,0])

    return raypos, segment_length, valid, tvals



def near_far_linear_ray_generation(campos,
                                   raydir,
                                   point_count,
                                   near=0.1,
                                   far=10,
                                   jitter=0.,
                                   **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # near: N x 1 x 1
    # far:  N x 1 x 1
    # jitter: float in [0, 1), a fraction of step length
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    # print("campos", campos.shape)
    # print("raydir", raydir.shape)
    tvals = torch.linspace(0, 1, point_count + 1,
                           device=campos.device).view(1, -1)
    tvals = near * (1 - tvals) + far * tvals  # N x 1 x Sammples
    segment_length = (tvals[..., 1:] -
                      tvals[..., :-1]) * (1 + jitter * (torch.rand(
                          (raydir.shape[0], raydir.shape[1], point_count),
                          device=campos.device) - 0.5))

    end_point_ts = torch.cumsum(segment_length, dim=2)
    end_point_ts = torch.cat([
        torch.zeros((end_point_ts.shape[0], end_point_ts.shape[1], 1),
                    device=end_point_ts.device), end_point_ts
    ],
                             dim=2)
    end_point_ts = near + end_point_ts

    middle_point_ts = (end_point_ts[:, :, :-1] + end_point_ts[:, :, 1:]) / 2
    raypos = campos[:, None, None, :] + raydir[:, :, None, :] * middle_point_ts[:, :, :, None]
    valid = torch.ones_like(middle_point_ts,
                            dtype=middle_point_ts.dtype,
                            device=middle_point_ts.device)

    segment_length*=torch.linalg.norm(raydir[..., None, :], axis=-1)
    return raypos, segment_length, valid, middle_point_ts



def refine_ray_generation(campos,
                               raydir,
                               point_count,
                               prev_ts,
                               prev_weights,
                               domain_size=1.,
                               jitter=0,
                               **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # point_count: int
    # prev_ts: N x Rays x PrevSamples
    # prev_weights: N x Rays x PrevSamples
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    with torch.no_grad():
        end_point_ts = sample_pdf(prev_ts[..., None], prev_weights,
                                  point_count + 1, jitter <= 0)
        end_point_ts = end_point_ts.view(end_point_ts.shape[:-1])
        segment_length = end_point_ts[:, :, 1:] - end_point_ts[:, :, :-1]
        middle_point_ts = (end_point_ts[:, :, :-1] +
                           end_point_ts[:, :, 1:]) / 2
        raypos = campos[:, None,
                        None, :] + raydir[:, :,
                                          None, :] * middle_point_ts[:, :, :,
                                                                     None]
        valid = torch.ones_like(middle_point_ts,
                                dtype=middle_point_ts.dtype,
                                device=middle_point_ts.device)
    segment_length*=torch.linalg.norm(raydir[..., None, :], axis=-1)
    return raypos, segment_length, valid, middle_point_ts


def nerf_refine_ray_generation(campos,
                               raydir,
                               point_count,
                               prev_ts,
                               prev_weights,
                               domain_size=1.,
                               jitter=0,
                               **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # point_count: int
    # prev_ts: N x Rays x PrevSamples,  uniformed depth segments between near and far
    # prev_weights: N x Rays x PrevSamples
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    with torch.no_grad():

        end_point_ts = sample_pdf(prev_ts[..., None], prev_weights,
                                  point_count + 1, jitter <= 0)
        end_point_ts = end_point_ts.view(end_point_ts.shape[:-1])
        segment_length = end_point_ts[:, :, 1:] - end_point_ts[:, :, :-1]
        segment_length *= torch.linalg.norm(raydir[..., None, :], axis=-1)
        middle_point_ts = (end_point_ts[:, :, :-1] +
                           end_point_ts[:, :, 1:]) / 2
        raypos = campos[:, None,
                        None, :] + raydir[:, :,
                                          None, :] * middle_point_ts[:, :, :,
                                                                     None]
        valid = torch.ones_like(middle_point_ts,
                                dtype=middle_point_ts.dtype,
                                device=middle_point_ts.device)

    return raypos, segment_length, valid, middle_point_ts


def refine_cube_ray_generation(campos,
                               raydir,
                               point_count,
                               prev_ts,
                               prev_weights,
                               domain_size=1.,
                               jitter=0,
                               **kargs):
    # inputs
    # campos: N x 3
    # raydir: N x Rays x 3, must be normalized
    # point_count: int
    # prev_ts: N x Rays x PrevSamples
    # prev_weights: N x Rays x PrevSamples
    # outputs
    # raypos: N x Rays x Samples x 3
    # segment_length: N x Rays x Samples
    # valid: N x Rays x Samples
    # ts: N x Rays x Samples
    with torch.no_grad():
        raypos, segment_length, _, middle_point_ts \
            = refine_ray_generation(campos,
                               raydir,
                               point_count,
                               prev_ts,
                               prev_weights,
                               domain_size=domain_size,
                               jitter=jitter,
                               **kargs)
        valid = torch.prod(torch.gt(raypos, -domain_size) *
                           torch.lt(raypos, domain_size),
                           dim=-1).byte()

    return raypos, segment_length, valid, middle_point_ts


def ray_march(ray_dist,
              ray_valid,
              ray_features,
              render_func,
              blend_func,
              bg_color=None):
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


    point_color = render_func(ray_features)

    # we are essentially predicting predict 1 - e^-sigma
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    # cumprod exclusive
    acc_transmission = torch.cumprod(1. - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1, )).to(
        opacity.device).float()  # N x R x 1

    background_transmission = acc_transmission[:, :, [-1]]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)

    blend_weight = blend_func(opacity, acc_transmission)[..., None]

    ray_color = torch.sum(point_color * blend_weight, dim=-2, keepdim=False)
    if bg_color is not None:
        ray_color += bg_color.to(opacity.device).float().view(
            background_transmission.shape[0], 1, 3) * background_transmission
    # #
    # if point_color.shape[1] > 0 and (torch.any(torch.isinf(point_color)) or torch.any(torch.isnan(point_color))):
    #     print("ray_color", torch.min(ray_color),torch.max(ray_color))

        # print("background_transmission", torch.min(background_transmission), torch.max(background_transmission))
    background_blend_weight = blend_func(1, background_transmission)
    # print("ray_color", torch.max(torch.abs(ray_color)), torch.max(torch.abs(sigma)), torch.max(torch.abs(opacity)),torch.max(torch.abs(acc_transmission)), torch.max(torch.abs(background_transmission)), torch.max(torch.abs(acc_transmission)), torch.max(torch.abs(background_blend_weight)))
    return ray_color, point_color, opacity, acc_transmission, blend_weight, \
        background_transmission, background_blend_weight


def alpha_ray_march(ray_dist, ray_valid, ray_features,
                    blend_func):
    sigma = ray_features[..., 0] * ray_valid.float()
    opacity = 1 - torch.exp(-sigma * ray_dist)

    acc_transmission = torch.cumprod(1. - opacity + 1e-10, dim=-1)
    temp = torch.ones(opacity.shape[0:2] + (1, )).to(
        opacity.device).float()  # N x R x 1
    background_transmission = acc_transmission[:, :, [-1]]
    acc_transmission = torch.cat([temp, acc_transmission[:, :, :-1]], dim=-1)

    blend_weight = blend_func(opacity, acc_transmission)[..., None]
    background_blend_weight = blend_func(1, background_transmission)

    return opacity, acc_transmission, blend_weight, \
        background_transmission, background_blend_weight
