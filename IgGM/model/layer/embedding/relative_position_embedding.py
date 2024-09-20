# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn


class RelativePositionEmbedding(nn.Module):
    """relative positional encoder."""

    def __init__(self, n_dims, ridx_max=32):
        super().__init__()
        self.n_dims = n_dims
        self.ridx_max = ridx_max
        self.linear = nn.Linear(self.ridx_max * 2 + 1, n_dims)

    def forward(self, aa_seq):
        """Get relative position encoding to update pair feature.

        Args:
            aa_seq: amino-acid sequence of length L

        Returns:
            z: initial pair feature
        """
        n_resds = len(aa_seq)
        device = self.linear.weight.device
        idxs_vec = torch.arange(n_resds).to(device)
        idxs_vec = idxs_vec.unsqueeze(dim=0)
        ridx_max = idxs_vec[:, None, :] - idxs_vec[:, :, None]
        ridx_max = ridx_max.clip(min=-self.ridx_max, max=self.ridx_max) + self.ridx_max
        onht_tns = torch.eye(self.ridx_max * 2 + 1, device=device)[ridx_max.long(), :].type_as(self.linear.weight)
        pfea_tns = self.linear(onht_tns)

        return pfea_tns
