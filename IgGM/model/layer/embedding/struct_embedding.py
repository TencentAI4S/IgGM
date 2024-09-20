"""Structure encoder.

Notes:
* This encoder utilitizes a similar strategy for extracting inter-residue pairwise information from
    per-atom 3D coordinates as in <RcEmbedNet>.
"""

import torch
from torch import nn

from IgGM.utils import cdist


class StructEncoder(nn.Module):
    """Structure encoder."""

    def __init__(self, n_dims):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims = n_dims

        # additional configurations
        self.n_bins = 18
        self.dist_min = 3.375
        self.dist_max = 21.375
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_bins

        # build an embedding layer
        self.embed = nn.Embedding(self.n_bins, self.n_dims)


    def forward(self, cord_tns, cmsk_tns):
        """Perform the forward pass.

        Args:
        * cord_tns: per-atom 3D coordinates of size N x L x M x 3
        * cmsk_tns: (optional) per-atom 3D coordinates's validness masks of size N x L x M

        Returns:
        * encd_tns: structure encodings of size N x L x L x D
        """
        if cord_tns.dim() == 3:
            cord_tns = cord_tns.unsqueeze(dim=0)
            cmsk_tns = cmsk_tns.unsqueeze(dim=0)
        # initialization
        n_smpls = cord_tns.shape[0]

        # calculate the pairwise distance between CA atoms
        dist_mat_list = []
        dmsk_mat_list = []
        for idx_smpl in range(n_smpls):
            cord_mat = cord_tns[idx_smpl, :, 1]  # CA is the second atom
            cmsk_vec = cmsk_tns[idx_smpl, :, 1]
            dist_mat = cdist(cord_mat)
            dmsk_mat = torch.outer(cmsk_vec, cmsk_vec)
            dist_mat_list.append(dist_mat)
            dmsk_mat_list.append(dmsk_mat)
        dist_tns = torch.stack(dist_mat_list, dim=0)  # N x L x L
        dmsk_tns = torch.stack(dmsk_mat_list, dim=0)  # N x L x L

        # build structure encodings
        idxs_tns = torch.clip(torch.floor(
            (dist_tns - self.dist_min) / self.bin_wid).to(torch.int64), 0, self.n_bins - 1)
        encd_tns = dmsk_tns.unsqueeze(dim=3) * self.embed(idxs_tns)

        return encd_tns
