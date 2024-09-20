# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import numpy as np
import torch
from torch import nn

from IgGM.model.layer import LayerNorm, Linear
from IgGM.transform import quat2rot, apply_trans


class InvariantPointAttention(nn.Module):
    """The invariant point attention (IPA) module.

    Args:
        c_s: number of dimensions in single features
        c_z: number of dimensions in pair features
        head_dim: number of dimensions in query/key/value embeddings
        num_heads: number of attention heads
        n_qpnts: number of points for query embeddings
        n_vpnts: number of points for value embeddings
        drop_prob: probability of an element to be zeroed (set to zero for DEQ models)
    """

    def __init__(
            self,
            c_s=384,
            c_z=256,
            head_dim=16,
            num_heads=12,
            n_qpnts=4,
            n_vpnts=8,
            drop_prob=0.1,
    ):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.n_dims_attn = head_dim
        self.n_heads = num_heads
        self.n_qpnts = n_qpnts
        self.n_vpnts = n_vpnts
        self.drop_prob = drop_prob
        self.n_dims_cord = 3  # DO NOT MODIFY!
        self.n_dims_shid = self.n_heads * \
                           (self.c_z + self.n_dims_attn + self.n_vpnts * 3 + self.n_vpnts)
        self.wc = np.sqrt(2.0 / (9.0 * self.n_qpnts))
        self.wl = np.sqrt(1.0 / 3.0)
        self.ws = np.log(np.exp(1.0) - 1.0)

        # sub-networks - Invariant Point Attention
        self.linear_q = Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_k = Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_v = Linear(self.c_s, self.n_heads * self.n_dims_attn, bias=False)
        self.linear_qp = Linear(
            self.c_s, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_kp = Linear(
            self.c_s, self.n_heads * self.n_qpnts * self.n_dims_cord, bias=False)
        self.linear_vp = Linear(
            self.c_s, self.n_heads * self.n_vpnts * self.n_dims_cord, bias=False)
        self.linear_b = Linear(self.c_z, self.n_heads, bias=False)
        self.linear_s = Linear(self.n_dims_shid, self.c_s)
        self.register_parameter(name='scale', param=nn.Parameter(self.ws * torch.ones((self.n_heads))))
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=2)

        # sub-networks - Feed-Forward Network
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.norm_1 = LayerNorm(self.c_s)
        self.mlp = nn.Sequential(
            Linear(self.c_s, self.c_s),
            nn.ReLU(),
            Linear(self.c_s, self.c_s),
            nn.ReLU(),
            Linear(self.c_s, self.c_s),
        )
        self.drop_2 = nn.Dropout(p=self.drop_prob)
        self.norm_2 = LayerNorm(self.c_s)

    def forward(self, s, z, quat_tns, trsl_tns):
        """
        Args:
            s: single features of size N x L x D_s
            z: pair features of size N x L x L x D_p
            quat_tns: quaternion vectors of size N x L x 4
            trsl_tns: translation vectors of size N x L x 3

        Returns:
            s: single features of size N x L x D_s
        """
        n_smpls, n_resds, _ = s.shape
        assert n_smpls == 1, f'batch size must be 1 in <InvPntAttn>; {n_smpls} detected'

        # calculate query/key/value embeddings
        q_tns = self.linear_q(s).view(n_smpls, n_resds, 1, self.n_heads, self.n_dims_attn)
        k_tns = self.linear_k(s).view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
        v_tns = self.linear_v(s).view(n_smpls, n_resds, self.n_heads, self.n_dims_attn)
        qp_tns = self.linear_qp(s).view(
            n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        kp_tns = self.linear_kp(s).view(
            n_smpls, n_resds, self.n_heads, self.n_qpnts, self.n_dims_cord)
        vp_tns = self.linear_vp(s).view(
            n_smpls, n_resds, self.n_heads, self.n_vpnts, self.n_dims_cord)
        b_tns = self.linear_b(z).view(n_smpls, n_resds, n_resds, self.n_heads)

        # apply global transformation on Q/K/V points
        rota_tns = quat2rot(quat_tns[0]).unsqueeze(dim=0)
        qp_tns_proj = apply_trans(qp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, 1, self.n_heads, self.n_qpnts, 3)
        kp_tns_proj = apply_trans(kp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, 1, n_resds, self.n_heads, self.n_qpnts, 3)
        vp_tns_proj = apply_trans(vp_tns, rota_tns, trsl_tns, grouped=True).view(
            n_smpls, n_resds, self.n_heads, self.n_vpnts, 3)

        # calculate the distance between query/key points
        dist_tns = torch.norm(qp_tns_proj - kp_tns_proj, dim=-1)  # N x L x L x H x P_q

        # compute attention weights
        qk_tns = torch.sum(q_tns * k_tns, dim=-1) / np.sqrt(self.n_dims_attn)  # N x L x L x H
        qkp_tns = 0.5 * self.wc * \
                  self.softplus(self.scale).view(1, 1, 1, -1) * torch.sum(dist_tns.square(), dim=-1)
        a_tns = self.softmax(self.wl * (qk_tns + b_tns - qkp_tns))  # N x L x L x H

        # update single features
        op_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            z.view(n_smpls, n_resds, n_resds, 1, self.c_z)
            , dim=2)  # N x L x H x D_p
        ov_tns = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            v_tns.view(n_smpls, 1, n_resds, self.n_heads, self.n_dims_attn)
            , dim=2)  # N x L x H x D_a
        ovp_tns_proj = torch.sum(
            a_tns.view(n_smpls, n_resds, n_resds, self.n_heads, 1) *
            vp_tns_proj.view(n_smpls, 1, n_resds, self.n_heads, self.n_vpnts * 3)
            , dim=2)  # N x L x H x (P_v x 3)
        ovp_tns = apply_trans(
            ovp_tns_proj, rota_tns, trsl_tns, grouped=True, reverse=True,
        ).view(n_smpls, n_resds, self.n_heads, self.n_vpnts * 3)  # N x L x H x (P_v x 3)
        ovp_tns_norm = torch.norm(
            ovp_tns.view(n_smpls, n_resds, self.n_heads, self.n_vpnts, 3)
            , dim=4)  # N x L x H x P_v
        shid_tns = torch.cat([op_tns, ov_tns, ovp_tns, ovp_tns_norm], dim=3)  # N x L x (H x D_h')
        s = s + self.linear_s(shid_tns.view(n_smpls, n_resds, self.n_dims_shid))

        # pass single features through a feed-forward network
        s = self.norm_1(self.drop_1(s))
        s = s + self.mlp(s)
        s = self.norm_2(self.drop_2(s))

        return s
