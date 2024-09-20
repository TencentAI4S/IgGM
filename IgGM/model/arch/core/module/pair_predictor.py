# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn

from IgGM.model.layer import get_activation_fn, Linear


# TODO: change to standord layernorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, pb_relax=False):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        self.pb_replax = pb_relax

    def forward(self, x):
        if self.pb_replax:
            x = x / (x.abs().max().detach() / 8)

        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class MLP(nn.Module):
    def __init__(self,
                 dim,
                 ffn_dim,
                 p_drop=0.1,
                 d_model_out=None,
                 activation='relu',
                 is_post_act_ln=False,
                 pb_relax=False
                 ):
        super(MLP, self).__init__()
        d_model_out = d_model_out or dim
        self.linear1 = Linear(dim, ffn_dim)
        self.post_act_ln = LayerNorm(ffn_dim, pb_relax=pb_relax) if is_post_act_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = Linear(ffn_dim, d_model_out)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.post_act_ln(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PairPredictor(nn.Module):
    """predict distance map from pair features
    """
    def __init__(self,
                 c_z,
                 bins=(37, 25, 25, 25),
                 is_use_ln=True,
                 pb_relax=False,
                 p_drop=0.0,
                 **kwargs):
        super(PairPredictor, self).__init__()
        self.c_z = c_z
        self.norm = LayerNorm(c_z, pb_relax=pb_relax) if is_use_ln else nn.Identity()
        self.proj = Linear(c_z, c_z)
        self.drop = nn.Dropout(p_drop)
        self.resnet_dist = MLP(c_z, ffn_dim=c_z * 4, d_model_out=bins[0], p_drop=p_drop, **kwargs)
        self.resnet_omega = MLP(c_z, ffn_dim=c_z * 4, d_model_out=bins[1], p_drop=p_drop,
                                **kwargs)
        self.resnet_theta = MLP(c_z, ffn_dim=c_z * 4, d_model_out=bins[2], p_drop=p_drop,
                                **kwargs)
        self.resnet_phi = MLP(c_z, ffn_dim=c_z * 4, d_model_out=bins[3], p_drop=p_drop,
                              **kwargs)

    def forward(self, z):
        """
        Args:
            z: [B, L, L, C], pair info
        """
        z = self.norm(z)
        z = self.drop(self.proj(z))
        # predict theta, phi (non-symmetric)
        logits_theta = self.resnet_theta(z).permute(0, 3, 1, 2)
        logits_phi = self.resnet_phi(z).permute(0, 3, 1, 2)
        # predict dist, omega
        z = 0.5 * (z + z.permute(0, 2, 1, 3))
        logits_dist = self.resnet_dist(z).permute(0, 3, 1, 2)
        logits_omega = self.resnet_omega(z).permute(0, 3, 1, 2)

        return logits_dist, logits_omega, logits_theta, logits_phi
