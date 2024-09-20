# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging

import numpy as np
import torch

from IgGM.transform import quat2rot
from IgGM.utils import cdist
from .parser import PdbParser
from .prot_constants import (RESD_NAMES_3C, RESD_MAP_1TO3, ATOM_NAMES_PER_RESD, ATOM_INFOS_PER_RESD,
                             ANGL_INFOS_PER_RESD, N_ATOMS_PER_RESD, N_ANGLS_PER_RESD)


class ProtStruct:
    """Protein structures (3D coordinates, local frames, and torsion angles)."""

    def __init__(self):
        self.aa_seq = None
        self.cord_tns = None  # L x M x 3 (full-atom 3D coordinates)
        self.cmsk_mat = None  # L x M

        # additional informations
        self.fram_tns_bb = None  # L x 1 x 4 x 3 (backbone local frames)
        self.fmsk_mat_bb = None  # L x 1
        self.fram_tns_sc = None  # L x K x 4 x 3 (side-chain local frames)
        self.fmsk_mat_sc = None  # L x K
        self.angl_tns = None  # L x K x 2 (torsion angles)
        self.amsk_mat = None  # L x K
        self.cmsk_mat_vld = None  # L x M (only depends on the sequence)
        self.cmsk_mat_sym = None  # L x M (only depends on the sequence)
        self.amsk_mat_sym = None  # L x K (only depends on the sequence)

        # data availability indicators
        self.has_cord_fa = False  # whether full-atom 3D coordinates are ready
        self.has_fram_bb = False  # whether backbone local frames are ready
        self.has_fram_sc = False  # whether side-chain local frames are ready
        self.has_angl = False  # whether torsion angles are ready
        self.has_mask = False  # whether valid/symmetric-or-not masks are ready

        # auxiliary constants
        self.eps = 1e-6

    def init_from_file(self, fas_fpath, pdb_fpath):
        """Initialize the protein structure from FASTA & PDB files.

        Args:
        * fas_fpath: path to the FASTA file
        * pdb_fpath: path to the PDB file

        Returns: n/a
        """

        # initialization
        self.aa_seq, self.cord_tns, self.cmsk_mat, _, error_msg = \
            PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'

        # update data availability indicators
        self.has_cord_fa = True  # even if the PDB file only contains CA atoms
        self.has_fram_bb = False
        self.has_fram_sc = False
        self.has_angl = False
        self.has_mask = False

    def init_from_cord(self, aa_seq, cord_tns, cmsk_mat):
        """Initialize the protein structure from 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

        Returns: n/a
        """

        # initialization
        self.aa_seq = aa_seq
        self.cord_tns = cord_tns
        self.cmsk_mat = cmsk_mat

        # update data availability indicators
        self.has_cord_fa = True  # even if <cord_tns> only contains CA atoms
        self.has_fram_bb = False
        self.has_fram_sc = False
        self.has_angl = False
        self.has_mask = False

    def init_from_param(self, aa_seq, params, converter, atom_set='fa'):
        """Initialize the protein structure from QTA parameters.

        Args:
        * aa_seq: amino-acid sequence
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
          > quat: per-residue quaternion vectors of size L x 4
          > trsl: per-residue translation vectors size size L x 3
          > angl: per-residue torsion angles of size L x K x 2
        * converter: <ProtConverter> object for coord-frame-angle conversions
        * atom_set: (optional) which atoms to be reconstructed (choices: 'ca' OR 'fa')

        Note:
        * We do not take any validness masks as inputs, since all the predicted QTA parameters are
            assumed to be valid.
        """

        # initialization
        n_resds = len(aa_seq)
        device = params['quat'].device

        # convert QTA parameters into backbone local frames & torsion angles
        self.aa_seq = aa_seq
        rota_tns = quat2rot(params['quat'])
        self.fram_tns_bb = torch.cat(
            [rota_tns, params['trsl'].unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        self.fmsk_mat_bb = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        self.angl_tns = \
            params['angl'] / (torch.norm(params['angl'], dim=2, keepdim=True) + self.eps)
        self.amsk_mat = torch.ones((n_resds, N_ANGLS_PER_RESD), dtype=torch.int8, device=device)

        # reconstruct per-atom 3D coordinates
        assert atom_set in ['ca', 'b3', 'b4', 'fa'], f'unrecognized atom set: {atom_set}'
        self.cord_tns, self.cmsk_mat = converter.fa2cord(
            self.aa_seq, self.fram_tns_bb, self.fmsk_mat_bb, self.angl_tns, self.amsk_mat, atom_set)

        # update data availability indicators
        self.has_cord_fa = (atom_set == 'fa')
        self.has_fram_bb = True
        self.has_fram_sc = False
        self.has_angl = True
        self.has_mask = False

    def build_fram_n_angl(self, converter, build_sc=False):
        """Build backbone and/or side-chain local frames and torison angles from 3D coordinates.

        Args:
        * converter: <ProtConverter> object for coord-frame-angle conversions
        * build_sc: (optional) whether to build side-chain local frames

        Returns: n/a
        """

        # check whether full-atom 3D coordinates are provided
        assert self.has_cord_fa, 'full-atom 3D coordinates must be provided in advance'

        # build backbone local frames & torsion angles
        if not (self.has_fram_bb and self.has_angl):
            self.fram_tns_bb, self.fmsk_mat_bb, self.angl_tns, self.amsk_mat = \
                converter.cord2fa(self.aa_seq, self.cord_tns, self.cmsk_mat)
            self.has_fram_bb = True
            self.has_angl = True

        # (optional) build side-chain local frames
        if build_sc and not self.has_fram_sc:
            self.fram_tns_sc, self.fmsk_mat_sc = \
                converter.cord2fram(self.aa_seq, self.cord_tns, self.cmsk_mat, fram_set='sc')
            self.has_fram_sc = True

    def build_mask(self):
        """Build valid/symmetric-or-not masks for atoms & torsion angles."""

        assert (self.aa_seq is not None) and (self.cord_tns is not None)

        self.cmsk_mat_vld = self.get_cmsk_vld(self.aa_seq, self.cord_tns.device)
        self.cmsk_mat_sym = self.get_cmsk_sym(self.aa_seq, self.cord_tns.device)
        self.amsk_mat_sym = self.get_amsk_sym(self.aa_seq, self.cord_tns.device)
        self.has_mask = True

    def build_alt_pose(self, converter):
        """Build the alternative pose by flipping all the symmetric torsion angles.

        Args:
        * converter: <ProtConverter> object for coord-frame-angle conversions

        Returns:
        * cord_tns: alternative 3D coordinates of size L x M x 3
        * angl_tns: alternative torsion angles of size L x K x 2
        * fram_tns_sc: alternative side-chain local frames of size L x K x 4 x 3
        """
        device = self.cord_tns.device
        # flip all the symmetric torsion angles
        amsk_mat_sym = self.get_amsk_sym(self.aa_seq, device)
        angl_tns = (1 - 2 * amsk_mat_sym).unsqueeze(dim=2) * self.angl_tns

        # reconstruct per-atom 3D coordinates w/ flipped symmetric torsion angles
        cord_tns, _ = converter.fa2cord(
            self.aa_seq, self.fram_tns_bb, self.fmsk_mat_bb, angl_tns, self.amsk_mat, atom_set='fa')

        # build side-chain local frames
        fram_tns_sc, _ = converter.cord2fram(self.aa_seq, cord_tns, self.cmsk_mat, fram_set='sc')

        return cord_tns, angl_tns, fram_tns_sc

    def rename_sym_atoms(self, cord_tns_ref, cmsk_mat_ref, converter):  # pylint: disable=too-many-locals
        """Rename symmetric ground-truth atoms.

        Args:
        * cord_tns_ref: reference protein structure's 3D coordinates of size L x M x 3
        * cmsk_mat_ref: reference protein structure's 3D coordinates' validness masks of size L x M
        * converter: <ProtConverter> object for coord-frame-angle conversions

        Returns: n/a

        Note:
        * There is at most one symmetric rigid-body group in each residue. Hence, we only need to
            maintain a single swap-or-not indicator per residue.
        """

        # initialization
        device = self.cord_tns.device
        n_resds = self.cord_tns.shape[0]

        # build the alternative pose of current structure
        cord_tns_alt, angl_tns_alt, fram_tns_sc_alt = self.build_alt_pose(converter)

        # calculate pairwise distance matrices
        dist_mat_ref = cdist(cord_tns_ref.view(-1, 3))  # (L x M) x (L x M)
        dist_mat_bsc = cdist(self.cord_tns.view(-1, 3))
        dist_mat_alt = cdist(cord_tns_alt.view(-1, 3))

        # get per-atom symmetric-or-not masks
        cmsk_mat_sym = self.get_cmsk_sym(self.aa_seq, device)

        # calculate dRMSD for basic & alternative query structures
        cmsk_mat_cmb = self.cmsk_mat * cmsk_mat_ref
        dmsk_mat = (cmsk_mat_cmb * cmsk_mat_sym).view(-1, 1) * \
                   (cmsk_mat_cmb * (1 - cmsk_mat_sym)).view(1, -1)
        drmsd_bsc = torch.sum(
            (dmsk_mat * torch.abs(dist_mat_bsc - dist_mat_ref)).view(n_resds, -1), dim=1)
        drmsd_alt = torch.sum(
            (dmsk_mat * torch.abs(dist_mat_alt - dist_mat_ref)).view(n_resds, -1), dim=1)

        # rename symmetric ground-truth atoms (and side-chain local frames, if provided)
        rmsk_vec = torch.less(drmsd_bsc, drmsd_alt)
        self.cord_tns = torch.where(rmsk_vec.view(-1, 1, 1), self.cord_tns, cord_tns_alt)
        self.angl_tns = torch.where(rmsk_vec.view(-1, 1, 1), self.angl_tns, angl_tns_alt)
        self.fram_tns_sc = torch.where(
            rmsk_vec.view(-1, 1, 1, 1), self.fram_tns_sc, fram_tns_sc_alt)

    def summarize(self):
        """Summarize the data availability of various aspect in the protein structure.

        Args: n/a

        Returns: n/a
        """

        # data availability
        logging.info('full-atom coordinates: %s', self.has_cord_fa)
        logging.info('backbone local frames: %s', self.has_fram_bb)
        logging.info('side-chain local frames: %s', self.has_fram_sc)
        logging.info('torsion angles: %s', self.has_angl)
        logging.info('atom/angle masks: %s', self.has_mask)

        # 3D coordinates
        logging.info('aa_seq: %s', self.aa_seq)
        logging.info('cord_tns: %s / %s', self.cord_tns.shape, self.cord_tns.dtype)
        logging.info('cmsk_mat: %s / %s', self.cmsk_mat.shape, self.cmsk_mat.dtype)

        # backbone local frames
        if self.has_fram_bb:
            logging.info('fram_tns_bb: %s / %s', self.fram_tns_bb.shape, self.fram_tns_bb.dtype)
            logging.info('fmsk_mat_bb: %s / %s', self.fmsk_mat_bb.shape, self.fmsk_mat_bb.dtype)

        # side-chain local frames
        if self.has_fram_sc:
            logging.info('fram_tns_sc: %s / %s', self.fram_tns_sc.shape, self.fram_tns_sc.dtype)
            logging.info('fmsk_mat_sc: %s / %s', self.fmsk_mat_sc.shape, self.fmsk_mat_sc.dtype)

        # torsion angles
        if self.has_angl:
            logging.info('angl_tns: %s / %s', self.angl_tns.shape, self.angl_tns.dtype)
            logging.info('amsk_mat: %s / %s', self.amsk_mat.shape, self.amsk_mat.dtype)

        # atom/angle masks
        if self.has_mask:
            logging.info('cmsk_mat_vld: %s / %s', self.cmsk_mat_vld.shape, self.cmsk_mat_vld.dtype)
            logging.info('cmsk_mat_sym: %s / %s', self.cmsk_mat_sym.shape, self.cmsk_mat_sym.dtype)
            logging.info('amsk_mat_sym: %s / %s', self.amsk_mat_sym.shape, self.amsk_mat_sym.dtype)

    @classmethod
    def get_atoms(cls, aa_seq, atom_tns_all, atom_names_sel):  # pylint: disable=too-many-locals
        """Get 3D coordinates or validness masks for selected atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names of length M'

        Returns:
        * atom_tns_sel: selected 3D coordinates (L x M' x 3) or validness masks (L x M')

        Note:
        * If only one atom name if provided, then the per-atom dimension is squeezed.
        """

        # use the specifically optimized implementation if only CA atoms are needed
        if atom_names_sel == ['CA']:
            return atom_tns_all[:, 1]  # CA atom is always the 2nd atom, as defined in constants.py

        # initialization
        device = atom_tns_all.device
        n_atoms = len(atom_names_sel)

        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks
        for resd_name in RESD_NAMES_3C:
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name]
            idxs_vec_np = np.zeros((n_atoms), dtype=np.int64)
            msks_vec_np = np.zeros((n_atoms), dtype=np.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel in atom_names_all:  # otherwise, keep zeros unchanged
                    idxs_vec_np[idx_atom_sel] = atom_names_all.index(atom_name_sel)
                    msks_vec_np[idx_atom_sel] = 1
            idxs_vec_dict[resd_name] = idxs_vec_np
            msks_vec_dict[resd_name] = msks_vec_np

        # determine the overall indexing tensor based on the amino-acid sequence
        resd_names_3c = [RESD_MAP_1TO3[resd_name_1c] for resd_name_1c in aa_seq]
        idxs_mat_full_np = np.stack([idxs_vec_dict[x] for x in resd_names_3c], axis=0)
        msks_mat_full_np = np.stack([msks_vec_dict[x] for x in resd_names_3c], axis=0)
        idxs_mat_full = torch.tensor(idxs_mat_full_np, dtype=torch.int64, device=device)  # L x M'
        msks_mat_full = torch.tensor(msks_mat_full_np, dtype=torch.int64, device=device)  # L x M'

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat_full * torch.gather(atom_tns_all, 1, idxs_mat_full)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat_full.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

        # squeeze the dimension if only one atom is selected
        if n_atoms == 1:
            atom_tns_sel.squeeze_(dim=1)

        return atom_tns_sel

    @classmethod
    def get_cmsk_vld(cls, aa_seq, device):
        """Get per-atom valid-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <cmsk_mat>

        Returns:
        * cmsk_mat: per-atom valid-or-not masks of size L x M
        """

        # initialization
        n_resds = len(aa_seq)
        atom_names_vld = ['N', 'CA', 'C']
        # generate per-atom valid-or-not masks
        cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8, device=device)
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            atom_names = ATOM_NAMES_PER_RESD[resd_name_3c]
            for atom_name in atom_names_vld:
                idx_atom = atom_names.index(atom_name)
                cmsk_mat[idx_resd, idx_atom] = 1

        return cmsk_mat

    @classmethod
    def get_cmsk_sym(cls, aa_seq, device):  # pylint: disable=too-many-locals
        """Get per-atom symmetric-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <cmsk_mat>

        Returns:
        * cmsk_mat: per-atom symmetric-or-not masks of size L x M
        """

        # initialization
        n_resds = len(aa_seq)

        # generate per-atom symmetric-or-not masks
        cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8, device=device)
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name_3c]
            atom_infos = ATOM_INFOS_PER_RESD[resd_name_3c]
            angl_infos = ANGL_INFOS_PER_RESD[resd_name_3c]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    atom_names_sel = [x[0] for x in atom_infos if x[1] == idx_angl + 3]
                    for atom_name in atom_names_sel:
                        idx_atom = atom_names_all.index(atom_name)
                        cmsk_mat[idx_resd, idx_atom] = 1

        return cmsk_mat

    @classmethod
    def get_amsk_sym(cls, aa_seq, device):
        """Get per-angle symmetric-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <amsk_mat>

        Returns:
        * amsk_mat: per-angle symmetric-or-not masks of size L x K
        """

        # initialization
        n_resds = len(aa_seq)

        # generate per-atom symmetric-or-not masks
        amsk_mat = torch.zeros((n_resds, N_ANGLS_PER_RESD), dtype=torch.int8, device=device)
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            angl_infos = ANGL_INFOS_PER_RESD[resd_name_3c]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    amsk_mat[idx_resd, idx_angl + 2] = 1

        return amsk_mat

    @classmethod
    def get_cb_cords(cls, aa_seq, cord_tns):
        """Get 3D coordinates of CB (CA for Glycine) atoms.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3

        Returns:
        * cord_mat: CB (CA for Glycine) atoms' 3D coordinates of size L x 3
        """

        mask_vec = torch.tensor([x == 'G' for x in aa_seq], device=cord_tns.device)
        cord_mat_ca = cls.get_atoms(aa_seq, cord_tns, ['CA'])
        cord_mat_cb = cls.get_atoms(aa_seq, cord_tns, ['CB'])
        cord_mat = torch.where(mask_vec.unsqueeze(dim=1), cord_mat_ca, cord_mat_cb)

        return cord_mat
