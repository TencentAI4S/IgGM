# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from functools import partialmethod
from typing import Optional, List

import torch
from torch import nn

from IgGM.model.layer import Linear, LayerNorm
from IgGM.model.utils.chunk_utils import chunk_layer
from IgGM.utils.tensor import permute_final_dims
from .gating_multihead_attention import GatedMultiheadAttention


class TriangleAttention(nn.Module):
    """
    Args:
        head_dim: Overall hidden channel dimension (not per-head)
        num_heads: Number of attention heads
    """

    def __init__(
            self,
            dim,
            num_heads: int = 4,
            head_dim=None,
            starting: bool = True,
            inf: float = 1e9
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must bed devided by heads({num_heads})"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or self.dim // num_heads
        self.num_heads = num_heads
        self.starting = starting
        self.inf = inf
        self.layer_norm = LayerNorm(self.dim)
        self.linear = Linear(self.dim, self.num_heads, bias=False, init='normal')
        self.mha = GatedMultiheadAttention(self.dim,
                                           head_dim=head_dim,
                                           num_heads=self.num_heads)

    @torch.jit.ignore
    def _chunk(self,
               x: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int
               ) -> torch.Tensor:
        mha_inputs = {
            'query': x,
            'biases': biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2])
        )

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None
                ) -> torch.Tensor:
        """
        Args:
            x: [*, I, J, C_in], the pair representation
            mask: None or [*, I, J]

        Returns:
            [*, I, J, C_in] output tensor
        """
        if not self.starting:
            x = x.transpose(-2, -3)
            if mask is not None:
                mask = mask.transpose(-1, -2)

        biases = []
        if mask is not None:
            # [*, I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
            biases.append(mask_bias)

        x = self.layer_norm(x)
        # [*, H, I, J]
        triangle_bias = permute_final_dims(self.linear(x), [2, 0, 1])
        # [*, 1, H, I, J]
        triangle_bias = triangle_bias.unsqueeze(-4)
        biases.append(triangle_bias)

        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size)
        else:
            x = self.mha(x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
