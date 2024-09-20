# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Optional

import torch
from torch import nn

from IgGM.model.layer import Linear, LayerNorm, GELU
from .multihead_attention import MultiheadAttention


class TransformerLayer(nn.Module):
    """Transformer layer."""

    def __init__(
            self,
            dim,
            num_heads,
            ffn_dim=None,
            use_crp_embeddings: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim or 4 * dim
        self.num_heads = num_heads
        self.use_crp_embeddings = use_crp_embeddings
        self.gelu = GELU('erf')
        self.self_attn = MultiheadAttention(
            self.dim,
            self.num_heads,
            use_crp_embeddings=self.use_crp_embeddings,
        )
        self.self_attn_layer_norm = LayerNorm(self.dim)
        self.fc1 = Linear(self.dim, self.ffn_dim)
        self.fc2 = Linear(self.ffn_dim, self.dim)
        self.final_layer_norm = LayerNorm(self.dim)

    def forward(
            self,
            x,
            asym_ids: Optional[torch.Tensor] = None,
            attn_bias: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            spad_mask: Optional[torch.Tensor] = None,
            need_head_weights: bool = False,
    ):
        """
        Args:
            embd_tns: sequence embeddings of size L x N x D
            asym_ids: (optional) asymmetric IDs of size N x L
            attn_bias: (optional) biases for attention weights of size N x H x L x L
            attn_mask: (optional) masks for attention weights of size N x H x L x L
            spad_mask: (optional) sequence padding masks of size N x L
            need_head_weights: (optional) whether to return per-head attention weights instead

        Returns:
            embd_tns: updated sequence embeddings of size L x N x D
            attn_weights: attention weights
              > if <need_head_weights> is False, then <attn_weights> is of size N x L x L
              > if <need_head_weights> is True, then <attn_weights> is of size H x N x L x L
        """
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(
            x,
            asym_ids=asym_ids,
            attn_bias=attn_bias,
            attn_mask=attn_mask,
            key_padding_mask=spad_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
        )
        x = residual + x
        # feed-forward network
        residual = x
        x = self.final_layer_norm(x)
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn_weights
