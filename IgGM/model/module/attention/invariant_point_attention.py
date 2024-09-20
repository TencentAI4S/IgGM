# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/11/1 21:52
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from IgGM.model.layer import Linear
from IgGM.transform.affine import Rigid
from IgGM.utils.tensor import permute_final_dims, flatten_final_dims


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class InvariantPointAttention(nn.Module):
    """
    Implements Algorithm 22.

    Args:
        c_s: Single representation channel dimension
        c_z: Pair representation channel dimension
        head_dim: Hidden channel dimension
        num_heads: Number of attention heads
        num_qk_points: Number of query/key points to generate
        num_v_points: Number of value points to generate
        inf: Large number used for attention masking
        eps: Small number used in angle resnet normalization
    """

    def __init__(
            self,
            c_s: int = 384,
            c_z: int = None,
            head_dim: int = None,
            num_heads: int = 12,
            num_qk_points: int = 4,
            num_v_points: int = 8,
            bias: bool = True,
            inf: float = 1e5,
            eps: float = 1e-8,
            pack_qkv: bool = False
    ):
        super(InvariantPointAttention, self).__init__()
        self.c_s = c_s
        self.c_z = 0 if c_z is None else c_z
        self.num_heads = num_heads
        self.head_dim = head_dim or self.c_s // self.num_heads
        self.num_qk_points = num_qk_points
        self.num_v_points = num_v_points
        self.inf = inf
        self.eps = eps
        self.bias = bias
        self.dim = self.head_dim * self.num_heads  # 192
        self.pack_qkv = pack_qkv
        if self.pack_qkv:
            self.linear_qkv = Linear(self.c_s, 3 * self.dim, bias=bias)

            self.linear_qkv_points = Linear(self.c_s,
                                            self.num_heads * (2 * self.num_qk_points + self.num_v_points) * 3)
        else:
            self.linear_q = Linear(self.c_s, self.dim, bias=bias)
            self.linear_kv = Linear(self.c_s, 2 * self.dim, bias=bias)

            self.linear_q_points = Linear(self.c_s, self.num_heads * self.num_qk_points * 3)
            hpkv = self.num_heads * (self.num_qk_points + self.num_v_points) * 3
            self.linear_kv_points = Linear(self.c_s, hpkv)

        self.head_weights = nn.Parameter(torch.zeros((num_heads)))
        ipa_point_weights_init_(self.head_weights)

        if self.c_z > 0:
            self.linear_b = Linear(self.c_z, self.num_heads)

        concat_out_dim = self.num_heads * (
                self.c_z + self.head_dim + self.num_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")

    def _project_qkv(self, s: torch.Tensor):
        """Generate scalar activations
        Args:
            s: [*, seq_len, c_s]

        Returns:
            q, k, v: [*, seq_len, num_heads, head_dim]
        """
        if self.pack_qkv:
            qkv = self.linear_qkv(s).split([self.dim, self.dim, self.dim], dim=-1)  # [*, seq_len, dim]
            q, k, v = [x.view(x.shape[:-1] + (self.num_heads, -1)) for x in qkv]
        else:
            # [*, seq_len, dim]
            q = self.linear_q(s)
            # [*, seq_len, num_heads, head_dim]
            q = q.view(q.shape[:-1] + (self.num_heads, -1))
            kv = self.linear_kv(s)
            kv = kv.view(kv.shape[:-1] + (self.num_heads, -1))
            k, v = torch.split(kv, self.head_dim, dim=-1)

        return q, k, v

    def _apply_points(self, pts, r: Rigid):
        """
        Args:
            pts: [*, seq_len, num_heads * num_pts * 3]
            r: [*, seq_len]

        Returns:
            transformed_pts: [*, seq_len, num_heads]
        """
        pts = torch.split(pts, pts.shape[-1] // 3, dim=-1)
        pts = torch.stack(pts, dim=-1)  # [*, seq_len, num_heads, num_pts, 3]
        pts = r[..., None].apply(pts)
        # [*, seq_len, num_heads, num_pts, 3]
        pts = pts.view(pts.shape[:-2] + (self.num_heads, -1, 3))

        return pts

    def _project_points(self,
                        s: torch.Tensor,
                        r: Rigid):
        """Generate point activations

        Args:
            s: [*, seq_len, c_s]
            r: [*, seq_len]

        Returns:
            q_pts, k_pts, v_pts: [*, seq_len, num_heads, num_q/k/v_pts, 3]
        """
        if self.pack_qkv:
            qkv_pts = self.linear_qkv_points(s)
            qkv_pts = self._apply_points(qkv_pts, r).to(s.dtype)
            q_pts, k_pts, v_pts = torch.split(
                qkv_pts, [self.num_qk_points, self.num_qk_points, self.num_v_points], dim=-2
            )
        else:
            # [*, N_res, H * P_q * 3]
            q_pts = self.linear_q_points(s)
            dtype = s.dtype
            q_pts = self._apply_points(q_pts, r).to(dtype)
            # [*, seq_len, H * (P_q + P_v) * 3]
            kv_pts = self.linear_kv_points(s)
            # [*, seq_len, H * (P_q + P_v), 3]
            kv_pts = self._apply_points(kv_pts, r).to(dtype)
            # [*, seq_len, H, P_q/P_v, 3]
            k_pts, v_pts = torch.split(
                kv_pts, [self.num_qk_points, self.num_v_points], dim=-2
            )
        return q_pts, k_pts, v_pts

    def _compute_pts_attn_weights(self, q_pts, k_pts):
        """compute distance weight of q_pts and k_pts"""
        # caculate distance of points
        d = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        attn_weights = sum(torch.unbind(d ** 2, dim=-1))  # [*, seq_len, seq_len, num_heads, num_pts]
        head_weights = F.softplus(self.head_weights).view(
            *((1,) * len(attn_weights.shape[:-2]) + (-1, 1))
        )
        attn_weights = attn_weights * head_weights * math.sqrt(1.0 / (self.num_qk_points * 9.0 / 2))
        attn_weights = torch.sum(attn_weights, dim=-1) * (-0.5)  # [*, seq_len, seq_len, num_heads]
        # [*, H, N_res, N_res]
        attn_weights = permute_final_dims(attn_weights, (2, 0, 1))

        return attn_weights

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor] = None,
            r: Rigid = None,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            s: [*, seq_len, C_s] single representation
            z: [*, seq_len, seq_len, C_z] pair representation
            r: [*, seq_len] transformation object
            mask: [*, seq_len] mask

        Returns:
            [*, N_res, C_s] single representation update
        """
        q, k, v = self._project_qkv(s)  # [*, seq_len, num_heads, head_dim]
        attn_weights = permute_final_dims(q, (1, 0, 2)) @ permute_final_dims(k, (1, 2, 0))
        # 3 is [s_atten_weights, z_atten_weights, q_atten_weights]
        attn_weights *= math.sqrt(1.0 / self.head_dim)  # [*, num_heads, seq_len, seq_len]

        if z is not None:
            pair_attn_weights = permute_final_dims(self.linear_b(z), (2, 0, 1))  # [*, num_heads, seq_len, seq_len]
            attn_weights += pair_attn_weights

        q_pts, k_pts, v_pts = self._project_points(s, r)  # [*, seq_len, num_heads, num_pts, 3]
        pt_attn_weights = self._compute_pts_attn_weights(q_pts, k_pts)
        attn_weights += pt_attn_weights

        num_attns = 3 if z is not None else 2
        attn_weights = attn_weights * math.sqrt(1.0 / num_attns)
        if mask is not None:
            square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            square_mask = self.inf * (square_mask - 1)
            attn_weights = attn_weights + square_mask.unsqueeze(-3).to(attn_weights.dtype)

        # TODO: change to scale dot attention
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(s)
        o = (attn_weights @ v.transpose(-2, -3)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)  # [*, seq_len, H * C_hidden]

        v_pts = permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(attn_weights[..., None, :, :, None] * v_pts, dim=-2)
        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        # r is float dtype
        o_pt = r[..., None, None].invert_apply(o_pt).to(o_pt.dtype)
        # [*, N_res, H * P_v]
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2)
        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)
        output = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm]

        if z is not None:
            o_pair = attn_weights.transpose(-2, -3) @ z
            o_pair = flatten_final_dims(o_pair, 2)
            output.append(o_pair)

        s = self.linear_out(torch.cat(output, dim=-1))  # [*, seq_len, c_s]
        return s
