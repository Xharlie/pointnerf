import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError('activation layer [{}] is not found'.format(activation_type))
    return nonlinearity_layer


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        #  norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, num_groups=16, affine=True)
    elif norm_type == 'layer':
        norm_layer = nn.LayerNorm
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(it):
            lr_l = 1.0 - max(0, it - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    elif opt.lr_policy == 'iter_exponential_decay':
        def lambda_rule(it):
            lr_l = pow(opt.lr_decay_exp, it / opt.lr_decay_iters)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif  opt.lr_policy == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs, eta_min=1e-7)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_xavier_multiplier(m, gain):
    if isinstance(m, nn.Conv1d):
        ksize = m.kernel_size[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose1d):
        ksize = m.kernel_size[0] // m.stride[0]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv2d):
        ksize = m.kernel_size[0] * m.kernel_size[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose2d):
        ksize = m.kernel_size[0] * m.kernel_size[1] // m.stride[0] // m.stride[1]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Conv3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.ConvTranspose3d):
        ksize = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] // m.stride[0] // m.stride[
            1] // m.stride[2]
        n1 = m.in_channels
        n2 = m.out_channels

        std = gain * np.sqrt(2.0 / ((n1 + n2) * ksize))
    elif isinstance(m, nn.Linear):
        n1 = m.in_features
        n2 = m.out_features

        std = gain * np.sqrt(2.0 / (n1 + n2))
    else:
        return None

    return std


def xavier_uniform_(m, gain):
    std = get_xavier_multiplier(m, gain)
    m.weight.data.uniform_(-std * np.sqrt(3.0), std * np.sqrt(3.0))


def init_weights(net, init_type='xavier_uniform', gain=1):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'xavier_uniform':
                xavier_uniform_(m, gain)
            elif init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

        # if classname == 'ConvTranspose2d':
        #     m.weight.data[:, :, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2]
        # elif classname == 'ConvTranspose3d':
        #     m.weight.data[:, :, 0::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 0::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 0::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 0::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 0::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 1::2, 0::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]
        #     m.weight.data[:, :, 1::2, 1::2, 1::2] = m.weight.data[:, :, 0::2, 0::2, 0::2]

    net.apply(init_func)


def init_seq(s, init_type='xavier_uniform'):
    '''initialize sequential model'''
    for a, b in zip(s[:-1], s[1:]):
        if isinstance(b, nn.ReLU):
            init_weights(a, init_type, nn.init.calculate_gain('relu'))
        elif isinstance(b, nn.LeakyReLU):
            init_weights(a, init_type, nn.init.calculate_gain('leaky_relu', b.negative_slope))
        else:
            init_weights(a, init_type)
    init_weights(s[-1])


def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts