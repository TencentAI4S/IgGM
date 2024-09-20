# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn


class ChainRelativePositionEmbedding(nn.Module):
    """Chain relative positional encoder (as in AlphaFold-multimer)."""

    def __init__(self, n_dims, ridx_max=32):
        super().__init__()
        self.n_dims = n_dims
        self.ridx_max = ridx_max
        self.linear = nn.Linear(2 * self.ridx_max + 3, self.n_dims)

    def forward(self, chn_infos, asym_id):
        """Get chain relative positional encodings to update pair features.

        Args:
        * chn_infos: list of (chn_type, chn_len) tuples
            asym_id: asymmetric ID of length $sum_{i} L_{i}$ (index starts from 1)

        Returns:
            pfea_tns: update term for pair features of size N x L x L x c_z

        Notes:
        * For position (i, j), the value is (i-j) clipped to [-k, k] and one-hotted.
        * We use an extra bin to indicate whether two residues comes from different chains.
        """
        device = self.linear.weight.device
        # pair features - same chain entity (chains with identical sequences)
        asym_mat = (asym_id.view(-1, 1) == asym_id.view(1, -1))  # L x L
        asym_tns = asym_mat.to(torch.float32).unsqueeze(dim=2)  # L x L x 1

        # pair features - relative positional encodings
        idxs_vec = torch.cat([torch.arange(seq_len, device=device) for _, seq_len in chn_infos], dim=0)
        ridx_mat = idxs_vec.view(-1, 1) - idxs_vec.view(1, -1)
        ridx_mat_clip = torch.clip(ridx_mat + self.ridx_max, min=0, max=(2 * self.ridx_max))
        ridx_mat_finl = torch.where(
            asym_mat, ridx_mat_clip, (2 * self.ridx_max + 1) * torch.ones_like(ridx_mat_clip))
        onht_tns = nn.functional.one_hot(ridx_mat_finl, num_classes=(2 * self.ridx_max + 2))

        # build the update term for pair features
        pfea_tns = self.linear(torch.cat([asym_tns, onht_tns], dim=2).to(self.linear.bias.dtype)).unsqueeze(dim=0)

        return pfea_tns
