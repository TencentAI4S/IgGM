# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/7/21 19:23
from typing import Tuple

import torch
import torch.nn.functional as F

from .math import acos_linear_extrapolation, hat, hat_inv


def so3_relative_angle(
        R1: torch.Tensor,
        R2: torch.Tensor,
        cos_angle: bool = False,
        cos_bound: float = 1e-4,
        eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)


def so3_rotation_angle(
        R: torch.Tensor,
        eps: float = 1e-4,
        cos_angle: bool = False,
        cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
            fac1[:, None, None] * skews
            # pyre-fixme[16]: `float` has no attribute `__getitem__`.
            + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def so3_log_map(
        R: torch.Tensor,
        eps: float = 1e-4,
        cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    dim1, dim2 = R.shape[-2:]
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot


# -----------------------add on--------------------#
def skew2vec(skew: torch.Tensor) -> torch.Tensor:
    vec = torch.zeros_like(skew[..., 0])
    vec[..., 0] = skew[..., 2, 1]
    vec[..., 1] = -skew[..., 2, 0]
    vec[..., 2] = skew[..., 1, 0]
    return vec


def vec2skew(vec: torch.Tensor) -> torch.Tensor:
    skew = torch.repeat_interleave(torch.zeros_like(vec).unsqueeze(-1), 3, dim=-1)
    skew[..., 2, 1] = vec[..., 0]
    skew[..., 2, 0] = -vec[..., 1]
    skew[..., 1, 0] = vec[..., 2]
    return skew - skew.transpose(-1, -2)


def orthogonalise_matrix(R):
    """Orthogonalise rotation/affine matrices

    Ideally, 3D rotation matrices should be orthogonal,
    however during creation, floating point errors can build up.
    We SVD decompose our matrix as in the ideal case S is a diagonal matrix of 1s
    We then round the values of S to [-1, 0, +1],
    making U @ S_rounded @ V.T an orthonormal matrix close to the original.
    """
    u, s, vh = torch.linalg.svd(R[..., :3, :3])
    return u @ torch.diag_embed(s.round()) @ vh


def so3_log_matrix(R: torch.Tensor):
    """We use atan2 instead of acos here dut to better numerical stability.
    it means we get nicer behaviour around 0 degrees. More effort to derive sin terms
    but as we're dealing with small angles a lot, the tradeoff is worth it.
    """
    skew_mat = (R - R.transpose(-1, -2))
    sk_vec = skew2vec(skew_mat)
    s_angle = (sk_vec).norm(p=2, dim=-1) / 2
    c_angle = (torch.einsum('...ii', R) - 1) / 2
    angle = torch.atan2(s_angle, c_angle)
    scale = (angle / (2 * s_angle))
    # if s_angle = 0, i.e. rotation by 0 or pi (180), we get NaNs
    # by definition, scale values are 0 if rotating by 0.
    # This also breaks down if rotating by pi, fix further down
    scale[angle == 0.0] = 0.0
    log_r_mat = scale[..., None, None] * skew_mat

    # Check for NaNs caused by 180deg rotations.
    nanlocs = log_r_mat[..., 0, 0].isnan()
    nanmats = R[nanlocs]
    # We need to use an alternative way of finding the logarithm for nanmats,
    # Use eigendecomposition to discover axis of rotation.
    # By definition, these are symmetric, so use eigh.
    # NOTE: linalg.eig() isn't in torch 1.8,
    #       and torch.eig() doesn't do batched matrices
    eigval, eigvec = torch.linalg.eigh(nanmats)
    # Final eigenvalue == 1, might be slightly off because floats, but other two are -ve.
    # this *should* just be the last column if the docs for eigh are true.
    nan_axes = eigvec[..., -1, :]
    nan_angle = angle[nanlocs]
    nan_skew = vec2skew(nan_angle[..., None] * nan_axes)
    log_r_mat[nanlocs] = nan_skew

    return log_r_mat


def so3_matrix_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Calculates the geodesic distance between two rotation matrices.
    """
    mul = R1.transpose(-1, -2) @ R2
    log_mul = so3_log_matrix(mul)
    out = log_mul.norm(p=2, dim=(-1, -2))
    return out


def so3_matrix_cosine_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Calculate the cosine distance between two (batched) rotation matrices
    """
    # rotation matrices have A^-1 = A_transpose
    # multiplying the inverse of m2 by m1 gives us a combined transform
    # corresponding to roting by m1 and then back by the *opposite* of m2.
    # if m1 == m2, m_comb = identity
    m_comb = R2.transpose(-1, -2) @ R1
    # Batch trace calculation
    tra = torch.einsum('...ii', m_comb)
    # The trace of a rotation matrix = 1+2*cos(a)
    # where a is the angle in the axis-angle representation
    # Cosine dist is defined as 1 - cos(a)
    out = 1 - (tra - 1) / 2
    return out


def so3_scale_matrix(R, scalars):
    '''Scale the magnitude of a rotation matrix,
    e.g. a 45 degree rotation scaled by a factor of 2 gives a 90 degree rotation.

    This is the same as taking matrix powers, but pytorch only supports integer exponents

    So instead, we take advantage of the properties of rotation matrices
    to calculate logarithms easily. and multiply instead.
    '''
    logs = so3_log_matrix(R)
    scaled_logs = logs * scalars[..., None, None]
    out = torch.matrix_exp(scaled_logs)

    return out


def _axis_angle_to_matrix(rot_axis: torch.Tensor, angle: torch.Tensor, eps=1e-12):
    """Generates a rotation matrix (3x3) from axis-angle

    Args:
        rot_axis: Axis to rotate around, defined as vector from origin.
        angle: rotation angle
    """
    rot_axis = F.normalize(rot_axis, p=2, eps=eps)
    sk_mats = vec2skew(rot_axis)
    log_rmats = sk_mats * angle[..., None]
    rot_mat = torch.matrix_exp(log_rmats)
    return orthogonalise_matrix(rot_mat)


def _matrix_to_axis_angle(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates axis and angle of rotation from a rotation matrix.
    returns angles in [0,pi] range.
    Args:
        R: rotation matrix.
    """
    eps = 1e-6
    log_mat = so3_log_matrix(R)
    skew_vec = skew2vec(log_mat)
    angle = skew_vec.norm(p=2, dim=-1, keepdim=True).clamp_min(eps).expand_as(skew_vec)
    axis = skew_vec / angle

    return axis, angle


def so3_lerp_matrix(rot_a: torch.Tensor,
                    rot_b: torch.Tensor,
                    weight: torch.Tensor) -> torch.Tensor:
    """Weighted interpolation between rot_a and rot_b"""
    # Treat rot_b = rot_a @ rot_c
    # rot_a^-1 @ rot_a = I
    # rot_a^-1 @ rot_b = rot_a^-1 @ rot_a @ rot_c = I @ rot_c
    # once we have rot_c, use axis-angle forms to lerp angle
    rot_c = rot_a.transpose(-1, -2) @ rot_b
    axis, angle = _matrix_to_axis_angle(rot_c)
    # once we have axis-angle forms, determine intermediate angles.
    i_angle = weight * angle
    rot_c_i = _axis_angle_to_matrix(axis, i_angle)
    return rot_a @ rot_c_i


def so3_bezier_matrix(*rots, weight):
    if len(rots) == 2:
        return so3_lerp_matrix(*rots, weight=weight)
    else:
        a = so3_bezier_matrix(rots[:-1], weight=weight)
        b = so3_bezier_matrix(rots[1:], weight=weight)
        return so3_lerp_matrix(a, b, weight=weight)
