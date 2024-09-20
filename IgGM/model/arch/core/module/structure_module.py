# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from torch import nn

from IgGM.protein import ProtStruct, ProtConverter, AtomMapper
from IgGM.protein.prot_constants import N_ATOMS_PER_RESD, N_ANGLS_PER_RESD
from IgGM.protein.utils import init_qta_params
from .head import PLDDTHead, FrameAngleHead
from .invariant_point_attention import InvariantPointAttention


class StructureModule(nn.Module):
    """The AlphaFold2 structure module."""
    """The AlphaFold2 structure module."""

    def __init__(
            self,
            n_lyrs=8,  # number of layers (all layers share the same set of parameters)
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_encd=64,  # number of dimensions in positional encodings
            pred_oxyg=False,  # whether to predict backbone oxygen atoms
            pred_schn=False,  # whether to predict side-chain torsion angles
    ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        self.n_lyrs = n_lyrs
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_encd = n_dims_encd
        self.pred_oxyg = pred_oxyg
        self.pred_schn = pred_schn

        # additional configurations
        self.atom_mapper = AtomMapper()
        self.prot_struct = ProtStruct()
        self.prot_converter = ProtConverter()
        self.atom_set = 'fa' if self.pred_schn else ('b4' if self.pred_oxyg else 'b3')

        # initial inputs
        self.net = nn.ModuleDict()
        self.net['norm_s'] = nn.LayerNorm(self.n_dims_sfea)
        self.net['norm_p'] = nn.LayerNorm(self.n_dims_pfea)
        self.net['linear_s'] = nn.Linear(self.n_dims_sfea, self.n_dims_sfea)

        # InvPntAttn - update single features
        self.net['ipa'] = InvariantPointAttention(
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
        )

        # FramAnglNet - predict backbone frames and/or side-chain torsion angles
        self.net['fa'] = FrameAngleHead(
            c_s=self.n_dims_sfea,
            n_dims_encd=self.n_dims_encd,
            decouple_angle=self.pred_schn,
        )

        # PLddtNet - predict lDDT-CA scores
        self.net['plddt'] = PLDDTHead(c_s=self.n_dims_sfea)

    def forward(
            self, aa_seqs, sfea_tns, pfea_tns, encd_tns,
            n_lyrs=-1, cord_tns_init=None, cmsk_tns_init=None, rmsk_vec_motf=None,
    ):  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        """Perform the forward pass.

        Args:
        * aa_seqs: amino-acid sequences (each of length L)
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * encd_tns: positional encodings of size N x L x D_e
        * n_lyrs: (optional) number of <AF2SMod> layers (-1: default number of layers)
        * cord_tns_init: (optional) initial per-atom 3D coordinates of size N x L x M x 3
        * cmsk_tns_init: (optional) initial per-atom 3D coordinates' validness masks of size N x L x M
        * rmsk_vec_motf: (optional) per-residue motif-or-not masks of size L

        Returns:
        * sfea_tns: updated single features of size N x L x D_s
        * cord_list: list of per-atom 3D coordinates of size N x L x M x 3, one per layer
        * param_list: list of QTA parameters, one per layer
          > quat: quaternion vectors of size N x L x 4
          > trsl: translation vectors of size N x L x 3
          > angl: torsion angle matrices of size N x L x K x 2
          > quat-u: update signal of quaternion vectors of size N x L x 4
        * plddt_list: list of pLDDT scores, one per layer
          > logit: raw classification logits of size N x L x 50
          > plddt-r: per-residue predicted lDDT-Ca scores of size N x L
          > plddt-c: full-chain predicted lDDT-Ca scores of size N
        * fram_tns_sc: final layer's side-chain local frames of size N x L x K x 4 x 3

        Note:
        * If <cord_tns_init> and <cmsk_tns_init> are provided as additional inputs, then QTA
            parameters will be initialized from them.
        * If <rmsk_vec_motf> is provided, then motif residues (whose <rmsk_vec> entries equals 1)
            will not be updated to ensure the consistency w/ the specified motif structure.
        * In <cord_list>, only the last entry contains full-atom 3D coordinates, while all the other
            entries only contain C-Alpha atoms' 3D coordinates.
        """

        # initialization
        n_smpls, n_resds, _ = sfea_tns.shape
        n_frams = n_smpls * n_resds
        dtype, device = sfea_tns.dtype, sfea_tns.device
        n_lyrs = self.n_lyrs if n_lyrs == -1 else n_lyrs
        assert all(len(x) == n_resds for x in aa_seqs)
        if rmsk_vec_motf is not None:
            assert (cord_tns_init is not None) and (cmsk_tns_init is not None)

        # pre-process single & pair features
        sfea_tns_init = self.net['norm_s'](sfea_tns)
        pfea_tns = self.net['norm_p'](pfea_tns)
        sfea_tns = self.net['linear_s'](sfea_tns_init)

        # initialize backbone local frames
        if (cord_tns_init is None) or (cmsk_tns_init is None):
            quat_tns_init, trsl_tns_init, _ = init_qta_params(n_smpls, n_resds, mode='black-hole')
            quat_tns_init = quat_tns_init.to(dtype).to(device)
            trsl_tns_init = trsl_tns_init.to(dtype).to(device)
        else:
            quat_tns_init, trsl_tns_init = \
                self.__init_fram_from_cord(aa_seqs, cord_tns_init, cmsk_tns_init)

        # perform multiple forward passes
        quat_tns = quat_tns_init.detach().clone()
        trsl_tns = trsl_tns_init.detach().clone()
        cord_list, param_list, plddt_list, fram_tns_sc = [], [], [], None
        for idx_lyr in range(n_lyrs):
            # perform a single forward pass
            quat_tns = quat_tns.detach()  # no gradient propagation
            sfea_tns = self.net['ipa'](sfea_tns, pfea_tns, quat_tns, trsl_tns)
            quat_tns, trsl_tns, angl_tns, quat_tns_upd = \
                self.net['fa'](aa_seqs, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns)
            plddt_dict = self.net['plddt'](sfea_tns.detach())

            # replace motif residues' local frames
            if rmsk_vec_motf is not None:
                quat_tns = quat_tns + rmsk_vec_motf.view(1, -1, 1) * (quat_tns_init - quat_tns)
                trsl_tns = trsl_tns + rmsk_vec_motf.view(1, -1, 1) * (trsl_tns_init - trsl_tns)

            # pack QTA parameters into a dict
            param_dict = {
                'quat': quat_tns,  # N x L x 4
                'trsl': trsl_tns,  # N x L x 3
                'angl': angl_tns,  # N x L x K x 2
                'quat-u': quat_tns_upd,  # N x L x 4
            }

            # reconstruct per-atom 3D coordinates and side-chain local frames
            aa_seq_flat = ''.join(aa_seqs)  # concatenate all the sequences into one
            param_dict_flat = {k: v.view(n_frams, *v.shape[2:]) for k, v in param_dict.items()}
            if idx_lyr < n_lyrs - 1:
                self.prot_struct.init_from_param(
                    aa_seq_flat, param_dict_flat, self.prot_converter, atom_set='ca')
                cord_tns = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
            else:
                self.prot_struct.init_from_param(
                    aa_seq_flat, param_dict_flat, self.prot_converter, atom_set=self.atom_set)
                cord_tns = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)
                if self.atom_set == 'fa':
                    self.prot_struct.build_fram_n_angl(self.prot_converter, build_sc=True)
                    fram_tns_sc = self.prot_struct.fram_tns_sc.view(n_smpls, n_resds, N_ANGLS_PER_RESD, 4, 3)

            # record predictions from the current layer
            cord_list.append(cord_tns)
            param_list.append(param_dict)
            plddt_list.append(plddt_dict)

        return sfea_tns, cord_list, param_list, plddt_list, fram_tns_sc

    def __init_fram_from_cord(self, aa_seqs, cord_tns, cmsk_tns):
        """Initialize backbone local frames from initial 3D coordinates."""

        # initialization
        n_smpls, n_resds, _, _ = cord_tns.shape

        # obtain 3D coordinates for backbone atoms (N - CA - C)
        cord_tns_bb_list = []
        cmsk_mat_bb_list = []
        for idx, aa_seq in enumerate(aa_seqs):
            cord_tns_bb = self.atom_mapper.run(
                aa_seq, cord_tns[idx], frmt_src='n14-tf', frmt_dst='n3')  # L x 3 x 3
            cmsk_mat_bb = self.atom_mapper.run(
                aa_seq, cmsk_tns[idx], frmt_src='n14-tf', frmt_dst='n3')  # L x 3
            cord_tns_bb_list.append(cord_tns_bb)
            cmsk_mat_bb_list.append(cmsk_mat_bb)
        cord_tns_bb = torch.stack(cord_tns_bb_list, dim=0)
        cmsk_tns_bb = torch.stack(cmsk_mat_bb_list, dim=0)

        # initialize backbone local frames from 3D coordinates
        quat_tns, trsl_tns, _ = init_qta_params(
            n_smpls, n_resds, mode='3d-cord', cord_tns=cord_tns_bb, cmsk_tns=cmsk_tns_bb)

        return quat_tns, trsl_tns
