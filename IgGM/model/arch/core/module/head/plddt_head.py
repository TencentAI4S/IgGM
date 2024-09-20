# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn


class PLDDTHead(nn.Module):
    """The network for predicting lDDT-Ca scores.

    Args:
        num_bins: number of bins for pLDDT-Ca predictions
    """

    def __init__(self, c_s=384, num_bins=50):
        super().__init__()
        self.c_s = c_s
        self.num_bins = num_bins
        self.bin_vals = (torch.arange(self.num_bins) + 0.5) / self.num_bins

        # per-residue lDDT-Ca predictions
        self.net = nn.ModuleDict()
        self.net['lddt'] = nn.Sequential(
            nn.LayerNorm(self.c_s),
            nn.Linear(self.c_s, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.num_bins),
        )
        self.net['sfmx'] = nn.Softmax(dim=2)

    def forward(self, s):
        """
        Args:
            s: single features of size N x L x D_s

        Returns:
            plddt_dict: dict of pLDDT predictions
        """
        dtype = s.dtype
        device = s.device

        # convert <self.bin_vals> into the correct data type & device
        self.bin_vals = self.bin_vals.to(dtype).to(device)

        # predict per-residue & full-chain lDDT-Ca scores
        logt_tns = self.net['lddt'](s)
        plddt_res = torch.sum(self.bin_vals.view(1, 1, -1) * self.net['sfmx'](logt_tns), dim=2)
        plddt_chn = torch.mean(plddt_res, dim=1)

        # pack all the pLDDT predictions into a dict
        plddt_dict = {
            'logit': logt_tns[0],  # L x 50
            'plddt-r': plddt_res[0],  # L
            'plddt-c': plddt_chn[0],  # scalar
        }

        return plddt_dict
