# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch

from .prot_constants import (
    RESD_MAP_1TO3,
    ATOM_NAMES_PER_RESD_N3,
    ATOM_NAMES_PER_RESD_N4,
    ATOM_NAMES_PER_RESD_N14_TF,
    ATOM_NAMES_PER_RESD_N14_AF,
    ATOM_NAMES_PER_RESD_N37,
)


class AtomMapper():
    """The atom mapper between different formats of per-residue atom name list.
    Notes:
        * List of supported formats:
          > ATOM_NAMES_PER_RESD_N3 ('n3'): 3 atoms per residue (N / CA / C)
          > ATOM_NAMES_PER_RESD_N4 ('n4'): 3 atoms per residue (N / CA / C / O)
          > ATOM_NAMES_PER_RESD_N14_TF ('n14-tf'): 14 atoms per residue (IgGM)
          > ATOM_NAMES_PER_RESD_N14_AF ('n14-af'): 14 atoms per residue (AlphaFold / OpenFold)
          > ATOM_NAMES_PER_RESD_N37 ('n37'): 37 atoms per residue
        * This function may have performance issue, espeically for CUDA tensors. USE WITH CAUTION!
    """

    def __init__(self):
        self.atom_frmt_dict = {
            'n3': ATOM_NAMES_PER_RESD_N3,
            'n4': ATOM_NAMES_PER_RESD_N4,
            'n14-tf': ATOM_NAMES_PER_RESD_N14_TF,
            'n14-af': ATOM_NAMES_PER_RESD_N14_AF,
            'n37': ATOM_NAMES_PER_RESD_N37,
        }

        # prepare a dict of atom index mappings between different formats
        self.aidx_map_dict = {}
        for frmt_src in self.atom_frmts:
            for frmt_dst in self.atom_frmts:
                if frmt_src == frmt_dst:
                    continue
                self.aidx_map_dict[(frmt_src, frmt_dst)] = self._build_residue_idx_map(frmt_src, frmt_dst)

    @property
    def atom_frmts(self):
        """Get a list of all the available atom formats."""
        return sorted(list(self.atom_frmt_dict.keys()))

    def run(self, aa_seq, atom_tns_src, frmt_src=None, frmt_dst=None):
        """Run the atom mapper.

        Args:
        * aa_seq: amino-acid sequence of length L
        * atom_tns_src: per-atom tensor in the source format of size L x M_i (x *)
        * frmt_src: (optional) source format of per-residue atom name list
        * frmt_dst: (optional) target format of per-residue atom name list

        Returns:
        * atom_tns_dst: per-atom tensor in the target format of size L x M_o (x *)
        """

        # force to use CPU for actual conversion
        if atom_tns_src.device != torch.device('cpu'):
            return self.run(
                aa_seq, atom_tns_src.cpu(), frmt_src, frmt_dst).to(atom_tns_src.device)

        # choose different atom mapping methods
        assert (frmt_src is not None) and (frmt_dst is not None)
        atom_tns_dst = self._run(aa_seq, atom_tns_src, frmt_src, frmt_dst)

        return atom_tns_dst

    def run_batch(self, aa_seqs, atom_tns_src, frmt_src=None, frmt_dst=None, method='v2'):
        """Run the atom mapper in the batch mode.

        Args:
            aa_seqs: batched amino-acid sequences, each of length L
            atom_tns_src: batched per-atom tensor in the source format of size N x L x M_i (x *)
            frmt_src: (optional) source format of per-residue atom name list
            frmt_dst: (optional) target format of per-residue atom name list
            method: (optional) atom mapping method (choices: 'v1' / 'v2')

        Returns:
            atom_tns_dst: batched per-atom tensor in the target format of size N x L x M_o (x *)
        """

        atom_tns_dst = self.run(
            ''.join(aa_seqs), atom_tns_src.view(-1, *atom_tns_src.shape[2:]),
            frmt_src=frmt_src, frmt_dst=frmt_dst
        )  # (N x L) x M_o (x *)
        atom_tns_dst = atom_tns_dst.view(*atom_tns_src.shape[:2], *atom_tns_dst.shape[1:])

        return atom_tns_dst

    def remap(self, aa_seq_src, atom_tns_src, aa_seq_dst):
        """Re-map the per-atom tensor between different amino-acid sequences.

        Args:
        * aa_seq_src: source amino-acid sequence of length L
        * atom_tns_src: source per-atom tensor in the 'n14-tf' format of size L x 14 (x *)
        * aa_seq_dst: target amino-acid sequence of length L

        Returns:
        * atom_tns_dst: target per-atom tensor in the 'n14-tf' format of size L x 14 (x *)

        Notes:
        * It is assumed that per-atom 3D coordinates are arranged in the 'n14-tf' format.
        """

        # force to use CPU for actual conversion
        if atom_tns_src.device != torch.device('cpu'):
            return self.remap(aa_seq_src, atom_tns_src.cpu(), aa_seq_dst).to(atom_tns_src.device)

        # re-map the per-atom tensor between different amino-acid sequences
        atom_tns_bb = self.run(aa_seq_src, atom_tns_src, frmt_src='n14-tf', frmt_dst='n3')
        atom_tns_dst = self.run(aa_seq_dst, atom_tns_bb, frmt_src='n3', frmt_dst='n14-tf')

        return atom_tns_dst

    def _run(self, aa_seq, atom_tns_src, frmt_src, frmt_dst):
        """Run the atom mapper - v2."""

        # initialization
        n_resds = len(aa_seq)
        dtype = atom_tns_src.dtype
        device = atom_tns_src.device
        assert frmt_src in self.atom_frmt_dict, f'unrecognized source format: {frmt_src}'
        assert frmt_dst in self.atom_frmt_dict, f'unrecognized target format: {frmt_dst}'
        n_atoms_src = int(frmt_src.split('-')[0][1:])
        n_atoms_dst = int(frmt_dst.split('-')[0][1:])
        assert atom_tns_src.shape[0] == n_resds
        assert atom_tns_src.shape[1] == n_atoms_src
        atom_tns_src_2d = atom_tns_src.view(n_resds * n_atoms_src, -1)
        atom_tns_dst_2d = torch.zeros(
            [n_resds * n_atoms_dst, atom_tns_src_2d.shape[1]], dtype=dtype, device=device)

        # generate the mapping between atom indices
        aidx_map = self.aidx_map_dict[(frmt_src, frmt_dst)]
        idxs_atom_src = []
        idxs_atom_dst = []
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            if resd_name_1c not in RESD_MAP_1TO3:  # skip non-standard residues
                continue
            idx_atom_base_src = idx_resd * n_atoms_src
            idx_atom_base_dst = idx_resd * n_atoms_dst
            idxs_atom_src.extend([idx_atom_base_src + x for x in aidx_map[resd_name_1c][0]])
            idxs_atom_dst.extend([idx_atom_base_dst + x for x in aidx_map[resd_name_1c][1]])

        # generate a per-atom tensor in the target format
        atom_tns_dst_2d[idxs_atom_dst] = atom_tns_src_2d[idxs_atom_src]
        atom_tns_dst = atom_tns_dst_2d.view(n_resds, n_atoms_dst, *atom_tns_src.shape[2:])

        return atom_tns_dst

    def _build_residue_idx_map(self, frmt_src, frmt_dst):
        """Build the atom index mapping between different formats."""

        aidx_map = {}
        for resd_name_1c, resd_name_3c in RESD_MAP_1TO3.items():
            idxs_atom_src = []
            idxs_atom_dst = []
            atom_names_src = self.atom_frmt_dict[frmt_src][resd_name_3c]
            atom_names_dst = self.atom_frmt_dict[frmt_dst][resd_name_3c]
            for idx_atom_dst, atom_name_dst in enumerate(atom_names_dst):
                if atom_name_dst not in atom_names_src:
                    continue
                idxs_atom_src.append(atom_names_src.index(atom_name_dst))
                idxs_atom_dst.append(idx_atom_dst)
            aidx_map[resd_name_1c] = (idxs_atom_src, idxs_atom_dst)

        return aidx_map
