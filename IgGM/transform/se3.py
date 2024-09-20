# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import numpy as np
import torch


def calc_rot_n_tsl(x1, x2, x3, eps=1e-6):
    """Calculate the rotation matrix & translation vector.

    Args:
        x1: 1st atom's 3D coordinate of size 3
        x2: 2nd atom's 3D coordinate of size 3
        x3: 3rd atom's 3D coordinate of size 3

    Returns:
        rot_mat: rotation matrix of size 3 x 3
        tsl_vec: translation vector of size 3

    Note:
    * <x2> is the origin point
    * <x3> - <x2> defines the direction of X-axis
    * <x1> lies in the X-Y plane

    Reference:
        Jumper et al., Highly accurate protein structure prediction with AlphaFold. Nature, 2021.
        - Supplementary information, Section 1.8.1, Algorithm 21.
    """
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / (torch.norm(v1) + eps)
    u2 = v2 - torch.inner(e1, v2) * e1
    e2 = u2 / (torch.norm(u2) + eps)
    e3 = torch.linalg.cross(e1, e2)
    rot_mat = torch.stack([e1, e2, e3], dim=0).permute(1, 0)
    tsl_vec = x2

    return rot_mat, tsl_vec


def calc_rot_n_tsl_batch(cord_tns):
    """Calculate rotation matrices & translation vectors in the batch mode.

    Args:
        cord_tns: 3D coordinates of size N x 3 x 3

    Returns:
        rot_tns: rotation matrices of size N x 3 x 3
        tsl_mat: translation vectors of size N x 3
    """

    eps = 1e-6
    x1, x2, x3 = [x.squeeze(dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    v1 = x3 - x2
    v2 = x1 - x2
    e1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)
    u2 = v2 - torch.sum(e1 * v2, dim=1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + eps)
    e3 = torch.linalg.cross(e1, e2, dim=1)
    rot_tns = torch.stack([e1, e2, e3], dim=1).permute(0, 2, 1)
    tsl_mat = x2

    return rot_tns, tsl_mat


def calc_dihd_angl_batch(cord_tns):
    """Calculate dihedral angles in the batch mode.

    Args:
        cord_tns: 3D coordinates of size N x 4 x 3

    Returns:
        rad_vec: dihedral angles (in radian, ranging from -pi to pi) of size N
    """

    eps = 1e-6
    x1, x2, x3, x4 = [x.squeeze(dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.linalg.cross(a1, a2, dim=1)
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)  # is this necessary?
    v2 = torch.linalg.cross(a2, a3, dim=1)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)  # is this necessary?
    n1 = torch.norm(v1, dim=1)
    n2 = torch.norm(v2, dim=1)
    sign = torch.sign(torch.sum(v1 * a3, dim=1))
    sign[sign == 0.0] = 1.0  # to avoid multiplication with zero
    rad_vec = sign * \
              torch.arccos(torch.clip(torch.sum(v1 * v2, dim=1) / (n1 * n2 + eps), -1.0, 1.0))

    return rad_vec


def quat2rot_impl(qr, qx, qy, qz):  # pylint: disable=too-many-locals
    """Convert decomposed quaternion vectors into rotation matrices - core implementation.

    Args:
        qr: 1st components in quaternion vectors
        qx: 2nd components in quaternion vectors
        qy: 3rd components in quaternion vectors
        qz: 4th components in quaternion vectors

    Returns:
        rot_mats: rotation matrices of size L x 3 x 3

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.4.1
    """

    # calculate intermediate results
    qrr = torch.square(qr)
    qxx = torch.square(qx)
    qyy = torch.square(qy)
    qzz = torch.square(qz)
    qrx = 2 * qr * qx
    qry = 2 * qr * qy
    qrz = 2 * qr * qz
    qxy = 2 * qx * qy
    qxz = 2 * qx * qz
    qyz = 2 * qy * qz

    # calculate each entry in the rotation matrix
    r11 = qrr + qxx - qyy - qzz
    r12 = qxy - qrz
    r13 = qxz + qry
    r21 = qxy + qrz
    r22 = qrr - qxx + qyy - qzz
    r23 = qyz - qrx
    r31 = qxz - qry
    r32 = qyz + qrx
    r33 = qrr - qxx - qyy + qzz

    # stack all the entries into rotation matrices
    rot_mats = torch.stack([
        torch.stack([r11, r12, r13], dim=1),
        torch.stack([r21, r22, r23], dim=1),
        torch.stack([r31, r32, r33], dim=1),
    ], dim=1)

    return rot_mats


def quat2rot_full(quat_vecs):
    """Convert full quaternion vectors into rotation matrices.

    Args:
        quat_vecs: quaternion vectors of size L x 4

    Returns:
        rot_mats: rotation matrices of size L x 3 x 3
    """

    # configurations
    eps = 1e-6

    # obtain normalized quaternion vectors
    quat_vecs_norm = quat_vecs / (torch.norm(quat_vecs, dim=1, keepdim=True) + eps)
    quat_vecs_flip = quat_vecs_norm * torch.sign(quat_vecs_norm[:, :1] + eps)  # qr: non-negative
    qr, qx, qy, qz = [x.squeeze(dim=1) for x in torch.split(quat_vecs_flip, 1, dim=1)]

    # convert decomposed quaternion vectors into rotation matrices
    rot_mats = quat2rot_impl(qr, qx, qy, qz)

    return rot_mats


def quat2rot_part(quat_vecs):
    """Convert partial quaternion vectors into rotation matrices.

    Args:
        quat_vecs: quaternion vectors of size L x 3

    Returns:
        rot_mats: rotation matrices of size L x 3 x 3
    """

    # obtain normalized quaternion vectors
    norm_vec = torch.sqrt(1.0 + torch.sum(torch.square(quat_vecs), dim=1))
    qr = 1.0 / norm_vec
    qx = quat_vecs[:, 0] / norm_vec
    qy = quat_vecs[:, 1] / norm_vec
    qz = quat_vecs[:, 2] / norm_vec

    # convert decomposed quaternion vectors into rotation matrices
    rot_mats = quat2rot_impl(qr, qx, qy, qz)

    return rot_mats


def quat2rot(quat_vecs):
    """Convert full / partial quaternion vectors into rotation matrices.

    Args:
        quat_vecs: quaternion vectors of size L x 4 (full) or L x 3 (part)

    Returns:
        rot_mats: rotation matrices of size L x 3 x 3
    """

    return quat2rot_full(quat_vecs) if quat_vecs.shape[1] == 4 else quat2rot_part(quat_vecs)


def rtax2rot(rtax_vecs, eps=1e-6):
    """Convert rotation axis vectors into rotation matrices.

    Args:
        rtax_vecs: rotation axis vectors of size L x 3

    Returns:
        rot_mats: rotation matrices of size L x 3 x 3

    Reference:
        Z. Zhang and O.D. Faugeras, Determining motion from 3D line segment matches.
        Image and Vision Computing, 1991.
    """
    dtype = rtax_vecs.dtype
    device = rtax_vecs.device
    n_smpls = rtax_vecs.shape[0]

    # obtain anti-symmetric matrices and rotation angles
    ra, rb, rc = [x.squeeze(dim=1) for x in torch.split(rtax_vecs, 1, dim=1)]
    rz = torch.zeros_like(ra)
    rtax_mats = torch.stack([
        torch.stack([rz, -rc, rb], dim=1),
        torch.stack([rc, rz, -ra], dim=1),
        torch.stack([-rb, ra, rz], dim=1),
    ], dim=1)  # L x 3 x 3
    angl_vals = torch.sqrt(torch.sum(torch.square(rtax_vecs), dim=1) + eps ** 2)
    f_vec = torch.sin(angl_vals) / angl_vals
    g_vec = (1.0 - torch.cos(angl_vals)) / torch.square(angl_vals)

    # obtain rotation matrices
    rot_mats = torch.eye(3, dtype=dtype, device=device).view(1, 3, 3).repeat(n_smpls, 1, 1) + \
               f_vec.view(n_smpls, 1, 1) * rtax_mats + \
               g_vec.view(n_smpls, 1, 1) * torch.matrix_power(rtax_mats, 2)

    return rot_mats


def rot2ypr(rot_mats, tol=1e-4, eps=1e-6):
    """Convert rotation matrices into yaw-pitch-roll angles.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3

    Returns:
        yaw: 1st components in Euler angles
        ptc: 2nd components in Euler angles
        rll: 3rd components in Euler angles
        tol: determine whether a degenerate case is encountered

    Reference:
        J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
        Technical report, 2020. - Section 2.5.1.
    """
    # extract entries from rotation matrices
    r11, _, r13 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 0], 1, dim=1)]
    r21, _, r23 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 1], 1, dim=1)]
    r31, r32, r33 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 2], 1, dim=1)]

    # recover pitch, yaw, and roll angles (naive implementation)
    ptc = torch.atan2(-r31, torch.sqrt(torch.square(r11) + torch.square(r21)))
    yaw = torch.where(
        torch.gt(torch.abs(torch.abs(ptc) - np.pi / 2.0), tol),
        torch.atan2(r21, r11 + eps),
        torch.where(torch.gt(ptc, 0.0), torch.atan2(r23, r13 + eps), torch.atan2(-r23, -r13 + eps)),
    )
    rll = torch.where(
        torch.gt(torch.abs(torch.abs(ptc) - np.pi / 2.0), tol),
        torch.atan2(r32, r33 + eps),
        torch.zeros_like(ptc),
    )

    return yaw, ptc, rll


def ypr2quat(yaw, ptc, rll):
    """Convert yaw-pitch-roll angles into quaternion vectors.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3

    Returns:
        qr: 1st components in quaternion vectors
        qx: 2nd components in quaternion vectors
        qy: 3rd components in quaternion vectors
        qz: 4th components in quaternion vectors

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.1.1.
    """

    # calculate normalized quaternion vectors
    ptc_s, ptc_c = torch.sin(ptc / 2.0), torch.cos(ptc / 2.0)
    yaw_s, yaw_c = torch.sin(yaw / 2.0), torch.cos(yaw / 2.0)
    rll_s, rll_c = torch.sin(rll / 2.0), torch.cos(rll / 2.0)
    qr = rll_c * ptc_c * yaw_c + rll_s * ptc_s * yaw_s
    qx = rll_s * ptc_c * yaw_c - rll_c * ptc_s * yaw_s
    qy = rll_c * ptc_s * yaw_c + rll_s * ptc_c * yaw_s
    qz = rll_c * ptc_c * yaw_s - rll_s * ptc_s * yaw_c

    return qr, qx, qy, qz


def rot2quat_full(rot_mats):
    """Convert rotation matrices into full quaternion vectors.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3

    Returns:
        quat_vecs: quaternion vectors of size L x 4
    """

    # configurations
    eps = 1e-6

    # convert rotation matrices into raw components in quaternion vectors
    yaw, ptc, rll = rot2ypr(rot_mats)
    qr, qx, qy, qz = ypr2quat(yaw, ptc, rll)

    # ensure that the 1st component is non-negative
    quat_vecs = torch.stack([qr, qx, qy, qz], dim=1)
    quat_vecs = quat_vecs * torch.sign(quat_vecs[:, :1] + eps)  # qr: non-negative

    return quat_vecs


def rot2quat_part(rot_mats, eps=1e-6):
    """Convert rotation matrices into partial quaternion vectors.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3

    Returns:
        quat_vecs: quaternion vectors of size L x 3
    """
    # convert rotation matrices into raw components in quaternion vectors
    yaw, ptc, rll = rot2ypr(rot_mats)
    qr, qx, qy, qz = ypr2quat(yaw, ptc, rll)

    # unnormalize quaternion vectors so that the first component is fixed to 1
    qa = qx / (qr + eps)
    qb = qy / (qr + eps)
    qc = qz / (qr + eps)
    quat_vecs = torch.stack([qa, qb, qc], dim=1)

    return quat_vecs


def rot2quat(rot_mats, quat_type='full'):
    """Convert rotation matrices into full / partial quaternion vectors.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3
        quat_type: type of quaternion vectors (choices: 'full' / 'part')

    Returns:
        quat_vecs: quaternion vectors of size L x 4 (full) or L x 3 (part)
    """

    return rot2quat_full(rot_mats) if quat_type == 'full' else rot2quat_part(rot_mats)


def rot2rtax(rot_mats):  # pylint: disable=too-many-locals
    """Convert rotation matrices into rotation axis vectors.

    Args:
        rot_mats: rotation matrices of size L x 3 x 3

    Returns:
        rtax_vecs: rotation axis vectors of size L x 3

    Reference:
        Z. Zhang and O.D. Faugeras, Determining motion from 3D line segment matches.
        Image and Vision Computing, 1991.
    """
    eps = 1e-6

    # extract entries from rotation matrices
    r11, r12, r13 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 0], 1, dim=1)]
    r21, r22, r23 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 1], 1, dim=1)]
    r31, r32, r33 = [x.squeeze(dim=1) for x in torch.split(rot_mats[:, 2], 1, dim=1)]

    # obtain rotation axis vectors
    angl_vals = torch.arccos(torch.clip((r11 + r22 + r33 - 1.0) / 2.0, min=-1.0, max=1.0))
    rx = (r32 - r23) / (2 * torch.sin(angl_vals) + eps)
    ry = (r13 - r31) / (2 * torch.sin(angl_vals) + eps)
    rz = (r21 - r12) / (2 * torch.sin(angl_vals) + eps)
    rtax_vecs = angl_vals.unsqueeze(dim=1) * torch.stack([rx, ry, rz], dim=1)

    # reduce the norm of rotation axis vectors w/ equivalence preserved
    norm_vec_old = torch.norm(rtax_vecs, dim=1)
    norm_vec_new = torch.remainder(norm_vec_old, 2 * np.pi)
    rtax_vecs = (norm_vec_new / (norm_vec_old + eps)).unsqueeze(dim=1) * rtax_vecs

    return rtax_vecs


def apply_trans(cord_tns_raw, rot_tns_raw, tsl_tns_raw, grouped=False, reverse=False):
    """Apply the global transformation on 3D coordinates.

    Args:
    * cord_tns_raw: 3D coordinates of size M x 3 (grouped: False) or L x M x 3 (grouped: True)
    * rot_tns_raw: rotation matrices of size L x 3 x 3
    * tsl_tns_raw: translation vectors of size L x 3

    Returns:
    * cord_tns_out: projected 3D coordinates of size L x M x 3

    Note:
    * If <grouped> is False, then <cord_tns_raw> should be of size M x 3 (or equivalent size) and
      each coordinate will be transformed multiple times, one per frame, resulting in output 3D
      coordinates of size L x M x 3.
    * If <grouped> is True, then <cord_tns_raw> should be of size L x M x 3 (or equivalent size) and
      each coordinate will be transformed only once (by the corresponding frame), resulting in
      output 3D coordinates of size L x M x 3. This is only useful when computing point components
      of attention affinities in the AlphaFold2's IPA module.
    * If <reverse> is False, then x' = R * x + t; otherwise, x' = R^(-1) * (x - t).
    """

    # re-organize the layout of input arguments
    rot_tns = rot_tns_raw.view(-1, 3, 3)  # L x 3 x 3
    n_frams = rot_tns.shape[0]  # number of local frames
    tsl_tns = tsl_tns_raw.view(n_frams, 3)  # L x 3
    if not grouped:
        cord_tns = torch.reshape(cord_tns_raw, [1, -1, 3])  # 1 x M x 3
    else:
        cord_tns = torch.reshape(cord_tns_raw, [n_frams, -1, 3])  # L x M x 3

    # apply the global transformation
    if not reverse:
        cord_tns_out = tsl_tns.unsqueeze(dim=1) + torch.sum(
            rot_tns.unsqueeze(dim=1) * cord_tns.unsqueeze(dim=2), dim=3)
    else:
        rot_tns_inv = rot_tns.permute(0, 2, 1).unsqueeze(dim=1)  # R x R^T = R^T x R = I
        cord_tns_out = torch.sum(
            rot_tns_inv * (cord_tns - tsl_tns.unsqueeze(dim=1)).unsqueeze(dim=2), dim=3)

    return cord_tns_out
