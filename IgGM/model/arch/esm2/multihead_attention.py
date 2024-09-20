"""Multi-head attention layer w/ chain relative positional encoding.

Notes:
* The original <MultiheadAttention> module in the <esm> repo has following configurations:
  > kdim: None
  > vdim: None
  > dropout: 0.0
  > bias: True
  > add_bias_kv: False
  > add_zero_attn: False
  > self_attention: False
  > encoder_decoder_attention: False
  > use_rotary_embeddings: False
  > use_crp_embeddings: False

* In ESM-2 models, only following configurations are explicitly specified:
  > add_bias_kv: False
  > add_zero_attn: False
  > use_rotary_embeddings: True (which is different from the default value)

* Thus, for simplicity, we only expose following configurations in <MultiHeadAttn>:
  > use_crp_embeddings: False
  and all the remaining configurations are kept the same as ESM-2 models.
"""
from typing import Optional
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rotary_embedding import RotaryEmbedding


class MultiheadAttention(nn.Module):
    """Multi-head attention layer w/ chain relative positional encoding."""

    def __init__(
            self,
            dim,
            num_heads,
            use_crp_embeddings: bool = False,
            dropout: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_crp_embeddings = use_crp_embeddings
        self.dropout = dropout
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, '<embed_dim> must be divisible by <num_heads>'
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.rot_emb = RotaryEmbedding(dim=self.head_dim)

    def forward(
            self,
            query,
            key: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            asym_ids: Optional[Tensor] = None,
            attn_bias: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            query: query embeddings of size L_q x N x D
            key: (optional) key embeddings of size L_kv x N x D
            value: (optional) value embeddings of size L_kv x N x D
            asym_ids: (optional) asymmetric IDs of size N x L_q (in this case, <L_q> equals <L_kv>)
            attn_bias: (optional) biases for attention weights of size N x H x L_q x L_kv
            attn_mask: (optional) masks for attention weights of size N x H x L_q x L_kv
            key_padding_mask: (optional) masks for padded keys of size N x L_kv
            need_weights: (optional) whether to return attention weights averaged over heads
            need_head_weights: (optional) whether to return per-head attention weights instead

        Returns:
            attn: multi-head attention embeddings of size L_q x N x D
            attn_weights: attention weights (only if <need_weights> or <need_head_weights> is True)
              > if <need_head_weights> is False, then <attn_weights> is of size N x L_q x L_kv
              > if <need_head_weights> is True, then <attn_weights> is of size H x N x L_q x L_kv

        Notes:
            For <attn_mask> & <kpad_mask>, only unmasked positions participate in the computation.
        """
        seq_len_q, batch_size, _ = query.shape
        key = key if key is not None else query
        value = value if value is not None else query
        assert key.shape[0] == value.shape[0]
        seq_len_kv = key.shape[0]
        assert not (self.use_crp_embeddings and (seq_len_q != seq_len_kv))

        # obtain query, key, and value embeddings
        q = self.q_proj(query) * self.scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # calculate raw attention weights
        if not self.use_crp_embeddings:
            q, k = self.rot_emb(q, k)
            attn_weights = torch.bmm(
                q, k.transpose(1, 2)).view(batch_size, self.num_heads, seq_len_q, seq_len_kv)
        else:
            attn_weights_raw = torch.bmm(
                q, k.transpose(1, 2)).view(batch_size, self.num_heads, seq_len_q, seq_len_kv)
            q, k = self.rot_emb(q, k)
            attn_weights_rpe = torch.bmm(
                q, k.transpose(1, 2)).view(batch_size, self.num_heads, seq_len_q, seq_len_kv)
            attn_weights_avg = torch.mean(attn_weights_rpe - attn_weights_raw, dim=(0, 1), keepdim=True)
            asym_mask = torch.eq(asym_ids.unsqueeze(dim=2), asym_ids.unsqueeze(dim=1))
            attn_weights = torch.where(
                asym_mask.unsqueeze(dim=1), attn_weights_rpe, attn_weights_raw + attn_weights_avg)

        # adjust attention weights w/ biases & masks
        if attn_bias is not None:
            attn_weights += attn_bias

        if attn_mask is not None:
            attn_weights.masked_fill_(attn_mask, float('-inf'))

        if key_padding_mask is not None:
            attn_weights.masked_fill_(key_padding_mask.view(batch_size, 1, 1, seq_len_kv), float('-inf'))

        # convert attention weights to normalized probabilities
        attn_probs_raw = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_probs_raw, p=self.dropout, training=self.training)
        attn_probs = attn_probs.view(batch_size * self.num_heads, seq_len_q, seq_len_kv)
        # obtain output embeddings
        attn = torch.bmm(attn_probs, v)
        attn = attn.transpose(0, 1).contiguous().view(seq_len_q, batch_size, self.dim)
        attn = self.out_proj(attn)

        # obtain attention weights (normalized probabilities, instead of raw logits)
        if need_head_weights:
            attn_weights = attn_probs_raw.transpose(0, 1)  # before dropout
        elif need_weights:
            attn_weights = torch.mean(attn_probs_raw, dim=1)
        else:
            attn_weights = None

        return attn, attn_weights
