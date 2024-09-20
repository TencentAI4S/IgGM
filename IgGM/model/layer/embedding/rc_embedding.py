"""The recycled embedding network for MSA features, pair features, and 3D structures."""

import torch
from torch import nn

from IgGM.protein.prot_constants import N_ATOMS_PER_RESD
from IgGM.utils import cdist



class RcEmbedNet(nn.Module):
    """The recycled embedding network for MSA features, pair features, and 3D structures."""

    def __init__(
            self,
            n_dims_mfea=384,  # number of dimensions in MSA features
            n_dims_pfea=256,  # number of dimensions in pair features
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_mfea = n_dims_mfea
        self.n_dims_pfea = n_dims_pfea

        # additional configurations
        self.n_bins = 18
        self.dist_min = 3.375
        self.dist_max = 21.375
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_bins

        # build the initial mapping for single features
        self.norm_m = nn.LayerNorm(self.n_dims_mfea)
        self.norm_p = nn.LayerNorm(self.n_dims_pfea)
        self.embed = nn.Embedding(self.n_bins, self.n_dims_pfea)

        # initialize model weights to zeros, so that pre-trained XFold models are unaffected
        nn.init.zeros_(self.norm_m.weight)
        nn.init.zeros_(self.norm_m.bias)
        nn.init.zeros_(self.norm_p.weight)
        nn.init.zeros_(self.norm_p.bias)
        nn.init.zeros_(self.embed.weight)


    def forward(self, mfea_tns, pfea_tns, rc_inputs=None):
        """Perform the forward pass.

        Args:
        * mfea_tns: MSA features of size N x K x L x D_m
        * pfea_tns: pair features of size N x L x L x D_p
        * rc_inputs: (optional) dict of additional inputs for recycling embeddings
          > sfea: single features of size N x L x D_m
          > pfea: pair features of size N x L x L x D_p
          > cord: per-atom 3D coordinates of size N x L x M x 3

        Returns:
        * mfea_tns: updated MSA features of size N x K x L x D_m
        * pfea_tns: updated pair features of size N x L x L x D_p
        """

        # initialization
        n_smpls, _, n_resds, _ = mfea_tns.shape
        dtype = mfea_tns.dtype  # for compatibility w/ half-precision inputs
        device = mfea_tns.device

        # initialize additional inputs for recycling embeddings
        if rc_inputs is None:
            rc_inputs = {
                'sfea': torch.zeros_like(mfea_tns[:, 0]),
                'pfea': torch.zeros_like(pfea_tns),
                'cord': torch.zeros((n_smpls, n_resds, N_ATOMS_PER_RESD, 3), device=device),
            }

        # calculate the pairwise distance between CA atoms
        dist_mat_list = []
        for idx_smpl in range(n_smpls):
            cord_mat = rc_inputs['cord'][idx_smpl, :, 1]  # CA is the second atom
            dist_mat = cdist(cord_mat.to(torch.float32)).to(dtype)  # cdist() requires FP32 inputs
            dist_mat_list.append(dist_mat)
        dist_tns = torch.stack(dist_mat_list, dim=0)  # N x L x L

        # calculate update terms for single features
        sfea_tns_rc = self.norm_m(rc_inputs['sfea'])

        # calculate update terms for pair features
        idxs_tns = torch.clip(torch.floor(
            (dist_tns - self.dist_min) / self.bin_wid).to(torch.int64), 0, self.n_bins - 1)
        pfea_tns_rc = self.norm_p(rc_inputs['pfea']) + self.embed(idxs_tns)

        # update MSA & pair features
        mfea_tns = torch.cat(
            [(mfea_tns[:, 0] + sfea_tns_rc).unsqueeze(dim=1), mfea_tns[:, 1:]], dim=1)
        pfea_tns = pfea_tns + pfea_tns_rc

        return mfea_tns, pfea_tns
