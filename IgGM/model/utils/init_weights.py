# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from IgGM.utils import Registry

INIT_METHOD_REGISTRY = Registry('init_method')


def _calculate_fan(linear_weight_shape, fan='fan_in'):
    fan_out, fan_in = linear_weight_shape

    if fan == 'fan_in':
        f = fan_in
    elif fan == 'fan_out':
        f = fan_out
    elif fan == 'fan_avg':
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError('Invalid fan option')

    return f


def trunc_normal_init_(weights, scale=1.0, fan='fan_in'):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = np.prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


@INIT_METHOD_REGISTRY.register('lecun')
def lecun_normal_init_(weights, bias=None):
    trunc_normal_init_(weights, scale=1.0)
    if bias is not None:
        nn.init.zeros_(bias)


@INIT_METHOD_REGISTRY.register('relu')
@INIT_METHOD_REGISTRY.register('he')
def he_normal_init_(weights, bias=None):
    trunc_normal_init_(weights, scale=2.0)
    if bias is not None:
        nn.init.zeros_(bias)


@INIT_METHOD_REGISTRY.register('normal')
def normal_init_(weights, bias=None):
    torch.nn.init.kaiming_normal_(weights, nonlinearity='linear')
    if bias is not None:
        nn.init.zeros_(bias)


@INIT_METHOD_REGISTRY.register('glorot')
def glorot_uniform_init_(weights, bias=None):
    nn.init.xavier_uniform_(weights, gain=1)
    if bias is not None:
        nn.init.zeros_(bias)


@INIT_METHOD_REGISTRY.register('zeros')
@INIT_METHOD_REGISTRY.register('final')
def final_init_(weights, bias=None):
    nn.init.zeros_(weights)
    if bias is not None:
        nn.init.zeros_(bias)


@INIT_METHOD_REGISTRY.register('gating')
def gating_init_(weights, bias=None):
    nn.init.zeros_(weights)
    if bias is not None:
        nn.init._no_grad_fill_(bias, 1.0)
