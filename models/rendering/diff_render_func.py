import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import format as fmt


def find_render_function(name):
    if name == 'radiance':
        return radiance_render
    elif name == 'white':
        return white_color
    raise RuntimeError('Unknown render function: ' + name)


def find_blend_function(name):
    if name == 'alpha':
        return alpha_blend
    elif name == 'alpha2':
        return alpha2_blend

    raise RuntimeError('Unknown blend function: ' + name)


def find_tone_map(name):
    if name == 'gamma':
        return simple_tone_map
    elif name == 'normalize':
        return normalize_tone_map
    elif name == 'off':
        return no_tone_map

    raise RuntimeError('Unknown blend function: ' + name)


def alpha_blend(opacity, acc_transmission):
    return opacity * acc_transmission


def alpha2_blend(opacity, acc_transmission):
    '''
    Consider a light collocated with the camera,
    multiply the transmission twice to simulate the light in a round trip
    '''
    return opacity * acc_transmission * acc_transmission


def radiance_render(ray_feature):
    return ray_feature[..., 1:4]


def white_color(ray_feature):
    albedo = ray_feature[..., 1:4].clamp(0., 1.)
    return torch.ones_like(albedo)


def simple_tone_map(color, gamma=2.2, exposure=1):
    return torch.pow(color * exposure + 1e-5, 1 / gamma).clamp_(0, 1)

def no_tone_map(color, gamma=2.2, exposure=1):
    # return color.clamp
    return color

def normalize_tone_map(color):
    color = F.normalize(color, dim=-1)
    # print(color)
    return color * 0.5 + 0.5
