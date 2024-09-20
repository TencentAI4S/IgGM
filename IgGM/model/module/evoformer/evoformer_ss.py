# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn

from IgGM.model.layer import (
    Linear,
    LayerNorm,
    DropoutRowwise,
    DropoutColumnwise
)
from .outer_product_mean import OuterProductMeanSS
from ..attention import (
    GatedMultiheadAttention,
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)


class SeqAttentionWithPairBias(nn.Module):
    """Sequence attention with pairwise biases.

    Args:
        c_s: number of dimensions in single features
        c_z: number of dimensions in pair features
        c_h: number of dimensions in query/key/value embeddings
        num_heads: number of attention heads
    """

    def __init__(self, c_s, c_z, hea_dim=None, num_heads=12):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)
        self.linear_z = Linear(self.c_z, self.num_heads, bias=False, init='normal')
        self.mha = GatedMultiheadAttention(self.c_s, head_dim=hea_dim, num_heads=self.num_heads)

    def forward(self, s, z):
        """
        Args:
            s: [N, L, c_s], single features
            z: [N, L, L, c_z]pair features

        Returns:
            s: [N, L, c_s], updated single features
        """

        s = self.layer_norm_s(s)
        z = self.layer_norm_z(z)
        bias = self.linear_z(z).permute(0, 3, 1, 2)
        s = self.mha(s, biases=[bias, ])

        return s


class Transition(nn.Module):
    """Transition module for both single & pair features."""

    def __init__(self, c, n=4):
        super().__init__()
        self.c = c
        self.n = n
        self.layer_norm = LayerNorm(self.c)
        self.linear_1 = Linear(self.c, self.n * self.c)
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c, c)

    def forward(self, x):
        """
        Args:
            x: [N, L, c] or [N, L, L, c], single/pair features

        Returns:
            x: updated single/pair features
        """
        x = self.layer_norm(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class EvoformerBlockSS(nn.Module):
    """The Evoformer block for single-sequence inputs."""

    def __init__(
            self,
            c_s: int = 384,
            c_z: int = 256,
            c_h_seq_att: int = 32,
            c_h_opm: int = 32,
            c_h_pair_mul: int = 128,
            c_h_pair_att: int = 32,
            num_heads_seq: int = 12,
            num_heads_pair: int = 8,
            dropout_seq: float = 0.15,
            dropout_pair: float = 0.25
    ):
        super().__init__()
        # single stack
        self.seq_att = SeqAttentionWithPairBias(c_s, c_z, hea_dim=c_h_seq_att, num_heads=num_heads_seq)
        self.seq_trans = Transition(c_s)
        # single to pair
        self.opm = OuterProductMeanSS(c_s, c_z, c_hidden=c_h_opm)
        # pair stack
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_hidden=c_h_pair_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_hidden=c_h_pair_mul)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, head_dim=c_h_pair_att, num_heads=num_heads_pair)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, head_dim=c_h_pair_att, num_heads=num_heads_pair)
        self.pair_trans = Transition(c_z)

        # dropout
        self.seq_dropout = DropoutRowwise(dropout_seq)
        self.pair_dropout_row = DropoutRowwise(dropout_pair)
        self.pair_dropout_col = DropoutColumnwise(dropout_pair)

    def forward(self, s, z, chunk_size=None):
        """
        Args:
            s: [N, L, c_s], single features
            z: [N, L, L, c_z], pair features

        Returns:
            s: [N, L, c_s], updated single features
            z: [N, L, L, c_z], updated pair features
        """
        # single stack
        s = s + self.seq_dropout(self.seq_att(s, z))
        s = s + self.seq_trans(s)

        # single to pair
        z = z + self.opm(s)

        # pair stack
        z = z + self.pair_dropout_row(self.tri_mul_out(z))
        z = z + self.pair_dropout_row(self.tri_mul_in(z))
        z = z + self.pair_dropout_row(self.tri_att_start(z, chunk_size=chunk_size))
        z = z + self.pair_dropout_col(self.tri_att_end(z, chunk_size=chunk_size))
        z = z + self.pair_trans(z)

        return s, z


class EvoformerStackSS(nn.Module):
    """Stacked EvoformerBlockSS layers.

    Args:
        c_s: number of dimensions in single features
        c_z: number of dimensions in pair features
        num_layers: number of EvoformerBlockSS layers
    """

    def __init__(
            self,
            c_s=384,
            c_z=256,
            num_layers=8,
            activation_checkpoint_fn=None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.c_s = c_s
        self.c_z = c_z
        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.activation_checkpoint = False
        self.blocks = nn.ModuleList([
            EvoformerBlockSS(self.c_s, self.c_z)
            for _ in range(self.num_layers)
        ])

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def forward(self, s, z, chunk_size=None):
        """
        Args:
            s: [N x L x c_s], single features
            z: pair features of size N x L x L x c_z

        Returns:
            s: updated single features of size N x L x c_s
            z: updated pair features of size N x L x L x c_z
        """
        for block in self.blocks:
            if not (self.training and self.activation_checkpoint):
                s, z = block(s, z, chunk_size)
            else:
                s, z = self.activation_checkpoint_fn(block, s, z, chunk_size)

        return s, z
