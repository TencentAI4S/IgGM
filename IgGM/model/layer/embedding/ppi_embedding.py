# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch.nn as nn

from .utils import contact_to_ppi
from ..linear import Linear


class PPIEmbedding(nn.Module):
    """The Contact embedding module.
    Args:
         c_s: number of dimensions in single features
    """

    def __init__(self, c_s):
        super().__init__()
        self.emb_init = Linear(1, c_s)
        self.proj_ppi_fea = Linear(c_s, c_s)

    def preprocess(self, ligand_feat, receptor_feat, ic_feat):
        """Function to prepare inputs for PPI_emb model

        ic_feat (1) ppi_feat: (N, L, 1): target residue in contact with other targets
                (2) contact_feat: (N, L, L, 1): target contact map
        """
        # prepare PPI data
        if ic_feat.dim() == 3:
            ppi_feat = ic_feat
        elif ic_feat.dim() == 4:
            ppi_feat = contact_to_ppi(ligand_feat, receptor_feat, ic_feat)
            ppi_feat = ppi_feat.to(ic_feat)

        return ppi_feat

    def forward(self, ligand_feat, receptor_feat, ppi_data):
        ppi_feat = self.preprocess(ligand_feat, receptor_feat, ppi_data)
        if ppi_feat.dtype != self.proj_ppi_fea.weight.dtype:
            ppi_feat = ppi_feat.to(self.proj_ppi_fea.weight.dtype)
        ppi_sfea_tns = self.proj_ppi_fea(self.emb_init(ppi_feat))

        return ppi_sfea_tns
