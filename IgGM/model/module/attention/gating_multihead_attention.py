# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from IgGM.config import get_config
from IgGM.model.layer import Linear
from IgGM.utils.tensor import permute_final_dims, flatten_final_dims

try:
    from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention

    logging.warning(f'using DS4Sci_EvoformerAttention')
except ImportError:
    DS4Sci_EvoformerAttention = None


class GatedMultiheadAttention(nn.Module):
    """Gating mutlihead attention in AlphaFold

    Args:
        c_q: Input dimension of query data
        c_k: Input dimension of key data
        c_v: Input dimension of value data
        num_heads: Number of attention heads
        gating: Whether the output should be gated using query data
    """

    def __init__(
            self,
            c_q: int,
            *,
            c_k: int = None,
            c_v: int = None,
            head_dim: int = None,
            num_heads: int = 8,
            gating: bool = True
    ):
        super(GatedMultiheadAttention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k or c_q
        self.c_v = c_v or c_q
        if head_dim is None:
            self.head_dim = self.c_q // num_heads
        else:
            self.head_dim = head_dim

        self.num_heads = num_heads
        self.gating = gating
        self.linear_q = Linear(self.c_q, self.head_dim * self.num_heads, bias=False, init='glorot')
        self.linear_k = Linear(self.c_k, self.head_dim * self.num_heads, bias=False, init='glorot')
        self.linear_v = Linear(self.c_v, self.head_dim * self.num_heads, bias=False, init='glorot')
        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(self.c_q, self.head_dim * self.num_heads, init='gating')

        self.linear_o = Linear(self.head_dim * self.num_heads, self.c_q, init='final')
        self.use_evo_attn = get_config().model.attention_mode == "evo_attn"

    def _project_qkv(self,
                     q: torch.Tensor,
                     k: torch.Tensor = None,
                     v: torch.Tensor = None
                     ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            q,k,v: [*, seq_len, dim], input features
        """
        k = q if k is None else k
        v = k if v is None else v

        # [*, Q/K/V, H * c]
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # [*, Q/K, num_heads, c]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        return q, k, v

    def _deepspeed_evo_attn(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            biases: List[torch.Tensor],
    ):
        """Compute attention using the DeepSpeed DS4Sci_EvoformerAttention kernel.

        Args:
            q: [*, Q, H, C_hidden] query data
            k: [*, K, H, C_hidden] key data
            v: [*, V, H, C_hidden] value data
            biases: List of biases that broadcast to [*, H, Q, K]
        """

        def reshape_dims(x):
            no_batch_dims = len(x.shape[:-3])
            if no_batch_dims < 2:
                return x.reshape(*((1,) * (2 - no_batch_dims) + x.shape))
            if no_batch_dims > 2:
                return x.reshape(*((x.shape[0], -1) + x.shape[-3:]))
            return x

        # Reshape tensors to match expected input shape [B, N, Q/K, H, C_hidden]
        # for DS4Sci_EvoformerAttention() by adding or flattening batch dims as needed.
        orig_shape = q.shape
        if len(orig_shape[:-3]) != 2:
            q = reshape_dims(q)
            k = reshape_dims(k)
            v = reshape_dims(v)
            biases = [reshape_dims(b) for b in biases]
        # requires inputs to be type bf16 or fp16
        orig_dtype = q.dtype
        if orig_dtype not in [torch.bfloat16, torch.float16]:
            o = DS4Sci_EvoformerAttention(q.to(dtype=torch.bfloat16),
                                          k.to(dtype=torch.bfloat16),
                                          v.to(dtype=torch.bfloat16),
                                          [b.to(dtype=torch.bfloat16) for b in biases])
            o = o.to(dtype=orig_dtype)
        else:
            o = DS4Sci_EvoformerAttention(q, k, v, biases)

        o = o.reshape(orig_shape)
        return o

    def _scale_dot_product_attention(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_biases: Optional[List[torch.Tensor]] = None,
            attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        if attn_mask is not None:
            attn_mask = attn_mask.bool()

        if attn_biases:
            attn_biases = [attn_biases, ] if isinstance(attn_biases, torch.Tensor) else attn_biases
            attn_bias = attn_biases[0]
            for b in attn_biases[1:]:
                attn_bias = attn_bias + b

            if attn_mask is not None:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))

            attn_bias = attn_bias.type_as(query)
            attn_mask = attn_bias

        query = query.transpose(-2, -3)  # [*, num_heads, seq_len, dim]
        key = key.transpose(-2, -3)
        value = value.transpose(-2, -3)
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        else:
            head_dim = query.shape[-1]
            scale_factor = head_dim ** -0.5
            query = scale_factor * query
            # [*, H, C_hidden, K]
            key = permute_final_dims(key, (1, 0))
            # [*, H, Q, K]
            scores = query @ key
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(~attn_mask.bool(), float('-inf'))
            else:
                scores += attn_mask
            scores = F.softmax(scores.float(), dim=-1).type_as(query)
            # [*, H, Q, C_hidden]
            y = scores @ value

        y = y.transpose(-2, -3)  # [*, seq_len, H, C_hidden]

        return y

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor = None,
            value: torch.Tensor = None,
            biases: Optional[List[torch.Tensor]] = None,
            attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [*, Lq, Cq] query data, sequence feature[bs, seq_len, c] or msa feature[bs, num_seqs, seq_len, c]
            key: [*, Lk, Ck] key data
            value: [*, Lk, Ck] value data
            biases: [*, num_heads, Lq, Lk] List of biases that broadcast to the shape
            attn_mask: [*, num_heads, Lq, Lk] attention mask

        Returns:
            [*, Lq, Cq] attention update
        """
        q, k, v = self._project_qkv(query, key, value)
        # remove none value
        if biases is None:
            biases = []

        biases = [b for b in biases if b is not None]
        # when two more biases
        use_ds_attn = DS4Sci_EvoformerAttention is not None and len(biases) <= 2 and attn_mask is None

        if use_ds_attn and self.use_evo_attn:
            y = self._deepspeed_evo_attn(q, k, v, biases)
        else:
            y = self._scale_dot_product_attention(q, k, v,
                                                  attn_biases=biases,
                                                  attn_mask=attn_mask)
        if self.gating:
            g = self.linear_g(query).sigmoid()
            g = g.view(g.shape[:-1] + (self.num_heads, -1))  # [*, Q, H, C_hidden]
            y = y * g

        y = flatten_final_dims(y, 2)  # [*, seq_len, num_heads * c_hidden]
        # [*, seq_len, c]
        y = self.linear_o(y)

        return y
