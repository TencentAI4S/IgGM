# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/7/21 19:35
from typing import Optional, Union, Tuple, List

import torch
import torch.nn.functional as F

from .math import _copysign
from .rotation_conversions import quaternion_to_matrix

Device = Union[str, torch.device]


def random_quaternions(
        batch_dims: Union[Tuple, List, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad=False
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        batch_dims: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (*, 4).
    """
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]

    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((*batch_dims, 4), dtype=dtype, device=device, requires_grad=requires_grad)
    s = o.norm(dim=-1, keepdim=True)
    o = o / _copysign(s, o[..., 0:1])

    return o


def random_rotation(
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations((1,), dtype, device, requires_grad)[0]


def random_rotations(
        batch_dims: Union[Tuple, List, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(batch_dims, dtype=dtype, device=device, requires_grad=requires_grad)
    return quaternion_to_matrix(quaternions)


def random_translations(
        batch_dims: Union[Tuple, List, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]
    return torch.randn((batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)


def random_quaternion_poses(
        batch_dims: Union[Tuple, List, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]
    quaternions = random_quaternions(batch_dims, dtype=dtype, device=device, requires_grad=requires_grad)
    translations = torch.randn((batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad)
    return torch.cat([quaternions, translations], dim=0)


def random_angles(
        batch_dims: Union[Tuple, List, int],
        eps=1e-12,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]

    angles = torch.randn((*batch_dims, 2), dtype=dtype, device=device, requires_grad=requires_grad)
    return F.normalize(angles, p=2, dim=-1, eps=eps)


def identity_matrix(
        batch_dims: Union[Tuple, List, int],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
) -> torch.Tensor:
    rots = torch.eye(
        3, dtype=dtype, device=device, requires_grad=requires_grad
    )
    rots = rots.view(*((1,) * len(batch_dims)), 3, 3)
    rots = rots.expand(*batch_dims, -1, -1)
    rots = rots.contiguous()

    return rots


def identity_quaternions(batch_dims: Union[Tuple, List, int],
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False):
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]

    # qr, qx, qy, qz
    quaternions = torch.zeros(
        (*batch_dims, 4),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )
    with torch.no_grad():
        quaternions[..., 0] = 1
    return quaternions


def identity_translatins(batch_dims: Union[Tuple, List, int],
                         dtype: Optional[torch.dtype] = None,
                         device: Optional[torch.device] = None,
                         requires_grad: bool = False):
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]
    return torch.zeros(
        (*batch_dims, 3),
        dtype=dtype,
        device=device,
        requires_grad=requires_grad
    )


def empty_angles(batch_dims: Union[Tuple, List, int],
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 requires_grad: bool = False):
    if isinstance(batch_dims, int):
        batch_dims = [batch_dims, ]

    return torch.cat([
        torch.ones((*batch_dims, 1), dtype=dtype, device=device, requires_grad=requires_grad),  # cosine
        torch.zeros((*batch_dims, 1), dtype=dtype, device=device, requires_grad=requires_grad),  # sine
    ], dim=-1)
