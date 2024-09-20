# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Union

import torch
from torch import nn

from IgGM.model.layer import LayerNorm
from .data import Alphabet
from .head import ContactPredictionHead, RobertaLMHead
from .transformer_layer import TransformerLayer


class ESM2(nn.Module):
    """ESM2 model."""

    def __init__(
            self,
            num_layers: int = 33,
            embed_dim: int = 1280,
            attention_heads: int = 20,
            alphabet: Union[Alphabet, str] = 'ESM-1b',
            token_dropout: bool = True,
            use_crp_embeddings: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, Alphabet):
            alphabet = Alphabet.from_architecture(alphabet)

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.use_crp_embeddings = use_crp_embeddings

        # build sub-modules
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    self.attention_heads,
                    use_crp_embeddings=self.use_crp_embeddings,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = LayerNorm(self.embed_dim)
        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )
        if self.use_crp_embeddings:
            self.asym_emb = nn.Embedding(2, self.attention_heads)  # M_ij = 0 / 1
            self.enty_emb = nn.Embedding(2, self.attention_heads)  # N_ij = 0 / 1

    def forward(
            self,
            tokens,
            asym_ids=None,
            enty_ids=None,
            repr_layers=[],
            need_head_weights=False,
            return_contacts=False,
    ):
        """
        Args:
            tokens: amino-acid tokens of size N x L
            asym_ids: (optional) asymmetric IDs of size N x L
            enty_ids: (optional) entity IDs of size N x L
            repr_layers: (optional) list of layer indices for extracting hidden representations
            need_head_weights: (optional) whether to return attention weights
            return_contacts: (optional) whether to return contact predictions
        
        Returns:
        * result: dict of output tensors
          > logits: classification logits of size N x L x C (C: number of tokens)
          > representations: dict of sequence embeddings of size N x L x D, indexed by layer indices
          > attentions: stacked per-layer attention weights of size N x M x H x L x L
          > contacts: contact predictions of size N x L' x L'

        Notes:
        * N: batch size (i.e., number of sequences)
        * L: sequence length, including 1 prepended & 1 appended tokens
        * L': sequence length, excluding 1 prepended & 1 appended tokens
        * M: number of Transformer layers
        * H: number of attention heads
        """
        batch_size, seq_len = tokens.shape

        # contact prediction requires attention weights as inputs
        if return_contacts:
            need_head_weights = True

        # obtain sequence embeddings from tokens
        embd_tns = self.embed_scale * self.embed_tokens(tokens)  # N x L x D

        # reset sequence embeddings corresponding to padded tokens to zeros
        spad_mask = tokens.eq(self.padding_idx)  # N x L
        embd_tns.masked_fill_(spad_mask.unsqueeze(dim=-1), 0.0)

        # scale sequence embeddings based on the ratio of masked tokens
        if self.token_dropout:
            mask_ratio_train = 0.15 * 0.8
            smsk_mask = tokens.eq(self.mask_idx)  # N x L
            mask_ratio_valid = torch.sum(smsk_mask, dim=1) / torch.sum(~spad_mask, dim=1).to(embd_tns)
            embd_tns.masked_fill_(smsk_mask.unsqueeze(dim=-1), 0.0)
            embd_tns = embd_tns * \
                       (1 - mask_ratio_train) / (1 - mask_ratio_valid).view(batch_size, 1, 1)

        # reset sequence padding masks to None (if possible)
        if not spad_mask.any():
            spad_mask = None

        # initialize the recorder for per-layer sequence embeddings & attention weights
        hidden_representations = {}
        repr_layers = set(repr_layers)
        if 0 in repr_layers:
            hidden_representations[0] = embd_tns  # N x L x D
        if need_head_weights:
            attn_weights_list = []

        # calculate biases for attention weights (shared by all layers)
        attn_bias = None
        if self.use_crp_embeddings:
            asym_mask = torch.eq(asym_ids.unsqueeze(dim=2), asym_ids.unsqueeze(dim=1))
            enty_mask = torch.eq(enty_ids.unsqueeze(dim=2), enty_ids.unsqueeze(dim=1))
            attn_bias_asym = self.asym_emb(asym_mask.to(torch.int64))
            attn_bias_enty = self.enty_emb(enty_mask.to(torch.int64))
            attn_bias = (attn_bias_asym + attn_bias_enty).permute(0, 3, 1, 2)

        # perform the forward pass through Transformer layers
        embd_tns = embd_tns.transpose(0, 1)
        for idx, layer in enumerate(self.layers):
            embd_tns, attn_weights = layer(
                embd_tns,
                asym_ids=asym_ids,
                attn_bias=attn_bias,
                spad_mask=spad_mask,
                need_head_weights=need_head_weights,
            )
            if (idx + 1) in repr_layers:
                hidden_representations[idx + 1] = embd_tns.transpose(0, 1)  # N x L x D
            if need_head_weights:
                attn_weights_list.append(attn_weights.transpose(1, 0))  # N x H x L x L

        # apply layer normalization on the final layer's sequence embeddings
        embd_tns = self.emb_layer_norm_after(embd_tns).transpose(0, 1)  # N x L x D
        if (idx + 1) in repr_layers:
            hidden_representations[idx + 1] = embd_tns  # overwrite previous results

        # predict token indices w/ RobertaLMHead
        pred_tns = self.lm_head(embd_tns)  # N x L x C

        # pack all the output tensors into a dict
        result = {'logits': pred_tns, 'representations': hidden_representations}
        if need_head_weights:
            attn_weights = torch.stack(attn_weights_list, dim=1)  # N x M x H x L x L
            if spad_mask is not None:
                ppad_mask = torch.maximum(spad_mask.unsqueeze(dim=2), spad_mask.unsqueeze(dim=1))
                attn_weights.masked_fill_(ppad_mask.view(batch_size, 1, 1, seq_len, seq_len), 0.0)
            result['attentions'] = attn_weights
            if return_contacts:
                contacts = self.contact_head(tokens, attn_weights)
                result['contacts'] = contacts

        return result
