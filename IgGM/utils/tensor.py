# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from functools import partial
from typing import List, Union, Tuple

import numpy as np
import torch


def to(tensor,
       device=None,
       dtype=None,
       non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to(t,
                                  device=device,
                                  dtype=dtype,
                                  non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to(value,
                                device=device,
                                dtype=dtype,
                                non_blocking=non_blocking)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device=device,
                         dtype=dtype,
                         non_blocking=non_blocking)
    elif isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, dtype=dtype, device=device)
    else:
        return tensor


def to_tensor(data, dtype=None, device=None):
    return to(data, dtype=dtype, device=device)


def to_device(tensor, device, non_blocking=False):
    return to(tensor, device, non_blocking=non_blocking)


def clone(data):
    """Clone the data object w/ gradients detached.

    Args:
        data: source data object (list / dict / torch.Tensor / etc.)

    Returns:
        data: cloned data object (list / dict / torch.Tensor / etc.)
    """

    if isinstance(data, list):
        return [clone(x) for x in data]
    if isinstance(data, dict):
        return {k: clone(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.detach().clone()

    return data


def permute_final_dims(tensor: torch.Tensor,
                       inds: Union[List, Tuple]):
    zero_index = -1 * len(inds)  # -2
    first_inds = list(range(len(tensor.shape[:zero_index])))
    orders = first_inds + [zero_index + i for i in inds]
    return tensor.permute(orders)


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))


def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


tensor_dict_map = partial(dict_map, leaf_type=torch.Tensor)


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        print(type(tree))
        raise ValueError('Not supported')


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def cdist(x1, x2=None):
    """Calculate the pairwise distance matrix.

    Args:
        x1: input tensor of size N x D or B x N x D
        x2: (optional) input tensor of size M x D or B x M x D

    Returns:
        dist_tns: pairwise distance of size N x M or B x N x M

    Note:
    * If <x2> is not provided, then pairwise distance will be computed within <x1>.
    * The matrix multiplication approach will not be used to avoid the numerical stability issue.
    """
    x2 = x1 if x2 is None else x2

    # recursively call if the batch dimension is missing
    if (x1.ndim == 2) and (x2.ndim == 2):
        return cdist(x1.unsqueeze(dim=0), x2.unsqueeze(dim=0))[0]

    # validate inputs
    assert (x1.ndim == 3) and (x2.ndim == 3)
    assert (x1.shape[0] == x2.shape[0]) and (x1.shape[2] == x2.shape[2])

    # calculate the pairwise distance matrix
    dist = torch.cdist(x1, x2, compute_mode='donot_use_mm_for_euclid_dist')

    return dist


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    # create batch dim index
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad, )
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)
