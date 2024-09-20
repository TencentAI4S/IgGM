# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from torch import nn

from .utils import gen_contact, ppi_to_contact


class ContactEmebedding(nn.Module):
    """The Contact embedding module.

    Args:
         c_z: number of dimensions of pairwise features
    """

    def __init__(self, c_z):
        super().__init__()
        self.emb_init = nn.Linear(1, c_z)
        self.proj_contact_fea = nn.Linear(c_z, c_z)

    def preprocess(self, ligand_feat, receptor_feat, ic_feat):
        """
        Function to prepare inputs for Contact_emb model

        ic_feat (1) ppi_feat: (N, L, 1): target residue in contact with other targets
                (2) contact_feat: (N, L, L, 1): target contact map
        """
        # prepare contact data
        if ic_feat.dim() == 3:
            contact_feat = ppi_to_contact(ligand_feat, receptor_feat, ic_feat).to(ic_feat.device)
            contact_feat = contact_feat.to(ic_feat.dtype)
        elif ic_feat.dim() == 4:
            contact_feat = gen_contact(ligand_feat, receptor_feat, ic_feat)

        return contact_feat

    def forward(self, ligand_feat, receptor_feat, contact_data):
        contact_feat = self.preprocess(ligand_feat, receptor_feat, contact_data)
        if contact_feat.dtype != self.proj_contact_fea.weight.dtype:
            contact_feat = contact_feat.to(self.proj_contact_fea.weight.dtype)
        contact_pfeat_tns = self.proj_contact_fea(self.emb_init(contact_feat))

        return contact_pfeat_tns
