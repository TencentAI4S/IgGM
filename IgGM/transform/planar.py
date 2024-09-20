# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/14 22:30
import torch
import torch.nn.functional as F


def compute_dihedral_angle(coords, eps=1e-6):
    """compute dihedral angles

    Args:
        coords: [*, 4, 3], 3D coordinates

    Returns:
        angles: [*] dihedral angles (in radian, ranging from -pi to pi)
    """
    x1, x2, x3, x4 = torch.unbind(coords, dim=-2)
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.cross(a1, a2, dim=1)
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = torch.cross(a2, a3, dim=1)
    v2 = F.normalize(v2, dim=-1, eps=eps)
    n1 = torch.norm(v1, dim=-1)
    n2 = torch.norm(v2, dim=-1)
    sign = torch.sign(torch.sum(v1 * a3, dim=-1))
    sign[sign == 0.0] = 1.0  # to avoid multiplication with zero
    rad_vec = sign * \
              torch.arccos(torch.clip(torch.sum(v1 * v2, dim=-1) / (n1 * n2 + eps), -1.0, 1.0))

    return rad_vec
