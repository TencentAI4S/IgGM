# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from collections import defaultdict

import numpy as np
import torch

from IgGM.transform import (calc_rot_n_tsl, calc_rot_n_tsl_batch, calc_dihd_angl_batch)
from . import prot_constants as constants
from .prot_struct import ProtStruct


class ProtConverter():
    """Protein structure converter."""

    def __init__(self, cpu_only=True):
        self.cpu_only = cpu_only
        # side-chain rigid-groups' names (padded according to ARG and LYS)
        self.rgrp_names_pad = ['omega', 'phi', 'psi', 'chi1', 'chi2', 'chi3', 'chi4']

        # convert per-atom 3D coordinates into torch.Tensor
        self.cord_dict = defaultdict(dict)
        for resd_name in constants.RESD_NAMES_3C:
            for atom_name, _, cord_vals in constants.ATOM_INFOS_PER_RESD[resd_name]:
                self.cord_dict[resd_name][atom_name] = torch.tensor(cord_vals, dtype=torch.float32)

        # build base transformations (angle = 0) for side-chain frames
        self.trans_dict_base = self.__build_trans_dict_base()

        # additional configurations
        self.eps = 1e-4

    def cord2fa(self, aa_seq, cord_tns, cmsk_mat):  # pylint: disable=too-many-locals
        """Convert 3D coordinates to backbone local frames & torsion angles.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: 3D coordinates of size L x M x 3
        * cmsk_mat: 3D coordinates' validness masks of size L x M

        Returns:
        * fram_tns_bb: backbone local frames of size L x 1 x 4 x 3
        * fmsk_mat_bb: backbone local frames' validness masks of size L x 1
        * angl_tns: torsion angles of size L x K x 2
        * amsk_mat: torsion angles' validness masks of size L x K
        """

        # initialization
        device = cord_tns.device
        n_resds = cord_tns.shape[0]

        # force to use CPU-based conversion routine (faster!)
        if self.cpu_only and device != torch.device('cpu'):
            outputs = self.cord2fa(aa_seq, cord_tns.cpu(), cmsk_mat.cpu())
            return [x.to(device) for x in outputs]

        # gather N/CA/C-atom 3D coordinates, denoted as x0, x1, and x2, respectively
        atom_names = ['N', 'CA', 'C']
        cord_tns_bb = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)  # L x 3 x 3
        cmsk_mat_bb = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)  # L x 3

        # compute backbone local frames from N/CA/C-atom 3D coordinates
        rot_tns, tsl_mat = calc_rot_n_tsl_batch(cord_tns_bb)
        fram_tns_bb = torch.cat([rot_tns, tsl_mat.unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        fmsk_mat_bb = torch.prod(cmsk_mat_bb, dim=1, keepdim=True).to(torch.int8)  # L x 1

        # get atoms defining all the dihedral angles
        cord_tns_dihd, cmsk_tns_dihd = self.__get_dihd_atoms(aa_seq, cord_tns, cmsk_mat)

        # compute torsion angles
        angl_vec = calc_dihd_angl_batch(cord_tns_dihd.view(-1, 4, 3))
        angl_tns = torch.stack([
            torch.cos(angl_vec).view(n_resds, constants.N_ANGLS_PER_RESD),
            torch.sin(angl_vec).view(n_resds, constants.N_ANGLS_PER_RESD),
        ], dim=2)  # L x K x 2
        amsk_mat = torch.prod(cmsk_tns_dihd, dim=2).to(torch.int8)  # L x K

        return fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat

    def fa2cord(self, aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat,
                atom_set='fa'):  # pylint: disable=too-many-arguments
        """Convert backbone local frames & torsion angles to 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * fram_tns_bb: backbone local frames of size L x 1 x 4 x 3
        * fmsk_mat_bb: backbone local frames' validness masks of size L x 1
        * angl_tns: torsion angles of size L x K x 2
        * amsk_mat: torsion angles' validness masks of size L x K
        * atom_set: (optional) atom set (choices: 'ca' / 'b3' / 'b4' / 'fa')

        Returns:
        * cord_tns: 3D coordinates of size L x M x 3
        * cmsk_mat: 3D coordinates' validness masks of size L x M
        """

        # take the shortcut if only CA atoms are considered
        assert atom_set in ['ca', 'b3', 'b4', 'fa'], f'unrecognized atom set: {atom_set}'
        if atom_set == 'ca':
            return self.__fa2cord_ca(fram_tns_bb, fmsk_mat_bb)

        # initialization
        device = fram_tns_bb.device
        n_resds = fram_tns_bb.shape[0]

        # force to use CPU-based conversion routine (faster!)
        if self.cpu_only and device != torch.device('cpu'):
            outputs = self.fa2cord(
                aa_seq, fram_tns_bb.cpu(), fmsk_mat_bb.cpu(), angl_tns.cpu(), amsk_mat.cpu())
            return [x.to(device) for x in outputs]

        # initialize 3D coordinates & validness masks
        cord_tns = torch.zeros((n_resds, constants.N_ATOMS_PER_RESD, 3), dtype=torch.float32, device=device)
        cmsk_mat = torch.zeros((n_resds, constants.N_ATOMS_PER_RESD), dtype=torch.int8, device=device)

        # enumerate over all the residue types
        for resd_name_1c, resd_name_3c in constants.RESD_MAP_1TO3.items():
            idxs = [x for x in range(n_resds) if aa_seq[x] == resd_name_1c]
            if len(idxs) == 0:
                continue
            cord_tns[idxs], cmsk_mat[idxs] = self.__fa2cord_fa(
                resd_name_3c, fram_tns_bb[idxs], fmsk_mat_bb[idxs], angl_tns[idxs], amsk_mat[idxs])

        return cord_tns, cmsk_mat

    def cord2fram(self, aa_seq, cord_tns, cmsk_mat, fram_set='sc'):  # pylint: disable=too-many-locals
        """Convert 3D coordinates to backbone and/or side-chain local frames.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: 3D coordinates of size L x M x 3
        * cmsk_mat: 3D coordinates' validness masks of size L x M
        * fram_set: (optional) frame set (choices: 'bb' / 'sc' / 'bs')

        Returns:
        * fram_tns: backbone and/or side-chain local frames of size L x F x 4 x 3
        * fmsk_mat: backbone and/or side-chain local frames' validness masks of size L x F

        Note:
        * The number of local frames per residue depends on the <fram_set> argument:
            F = 1 (bb) / K (sc) / 1 + K (bs)
        """

        # initialization
        device = cord_tns.device
        n_resds = cord_tns.shape[0]
        build_bb = (fram_set in ['bb', 'bs'])
        build_sc = (fram_set in ['sc', 'bs'])

        # force to use CPU-based conversion routine (faster!)
        if self.cpu_only and device != torch.device('cpu'):
            outputs = self.cord2fram(aa_seq, cord_tns.cpu(), cmsk_mat.cpu())
            return [x.to(device) for x in outputs]

        # build backbone local frames
        if build_bb:
            atom_names = ['N', 'CA', 'C']
            cord_tns_bb = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)  # L x 3 x 3
            cmsk_mat_bb = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)  # L x 3
            rot_tns, tsl_mat = calc_rot_n_tsl_batch(cord_tns_bb)
            fram_tns_bb = torch.cat([rot_tns, tsl_mat.unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
            fmsk_mat_bb = torch.prod(cmsk_mat_bb, dim=1, keepdim=True).to(torch.int8)  # L x 1

        # build side-chain local frames
        if build_sc:
            cord_tns_dihd, cmsk_tns_dihd = self.__get_dihd_atoms(aa_seq, cord_tns, cmsk_mat)
            cord_tns_sc = torch.stack([
                cord_tns_dihd[:, :, 3],
                cord_tns_dihd[:, :, 2],
                cord_tns_dihd[:, :, 2] * 2 - cord_tns_dihd[:, :, 1],
            ], dim=2)
            rot_tns, tsl_mat = calc_rot_n_tsl_batch(cord_tns_sc.view(-1, 3, 3))
            fram_tns_sc = torch.cat([
                rot_tns.view(n_resds, constants.N_ANGLS_PER_RESD, 3, 3),
                tsl_mat.view(n_resds, constants.N_ANGLS_PER_RESD, 1, 3),
            ], dim=2)  # L x K x 4 x 3
            fmsk_mat_sc = torch.prod(cmsk_tns_dihd[:, :, 1:], dim=2).to(torch.int8)  # L x K

        # determine final outputs
        if fram_set == 'bb':
            fram_tns, fmsk_mat = fram_tns_bb, fmsk_mat_bb
        elif fram_set == 'sc':
            fram_tns, fmsk_mat = fram_tns_sc, fmsk_mat_sc
        elif fram_set == 'bs':
            fram_tns = torch.cat([fram_tns_bb, fram_tns_sc], dim=1)
            fmsk_mat = torch.cat([fmsk_mat_bb, fmsk_mat_sc], dim=1)
        else:
            raise ValueError(f'unrecognized frame set: {fram_set}')

        return fram_tns, fmsk_mat

    def __build_trans_dict_base(self):  # pylint: disable=too-many-locals
        """Build base transformations (angle = 0) for side-chain frames."""

        trans_dict_full = {}
        for resd_name in constants.RESD_NAMES_3C:
            # initialization
            trans_dict = {}
            atom_infos = constants.ATOM_INFOS_PER_RESD[resd_name]  # list of (atom name, RG index, cord.)
            angl_infos = constants.ANGL_INFOS_PER_RESD[resd_name]  # list of (angl name, symm, atom names)
            n_angls = len(angl_infos)

            # initialize backbone atoms' 3D coordinates w.r.t. the backbone frame
            cord_dict = {}
            for atom_name, idx_rgrp, _ in atom_infos:
                if idx_rgrp == 0:  # backbone rigid-group
                    cord_dict[atom_name] = self.cord_dict[resd_name][atom_name]

            # build the pre-omega to backbone transformation
            trans_dict['omega-bb'] = (
                torch.eye(3, dtype=torch.float32),
                torch.zeros((3), dtype=torch.float32),
            )  # identity mapping

            # build the phi to backbone transformation
            x1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
            x2 = cord_dict['N']
            x3 = cord_dict['CA']
            rot_mat, tsl_vec = calc_rot_n_tsl(x1, x2, x2 + (x2 - x3))
            trans_dict['phi-bb'] = (rot_mat, tsl_vec)

            # build psi & chiX to backbone transformations
            for idx_angl, (angl_name, _, atom_names_sel) in enumerate(angl_infos):
                # compute the rotation matrix and translation vector
                x1 = cord_dict[atom_names_sel[0]]
                x2 = cord_dict[atom_names_sel[1]]
                x3 = cord_dict[atom_names_sel[2]]
                rot_mat, tsl_vec = calc_rot_n_tsl(x1, x3, x3 + (x3 - x2))
                trans_dict[f'{angl_name}-bb'] = (rot_mat, tsl_vec)

                # transform all the atoms in the current rigid-group to the backbone frame
                for atom_name, idx_rgrp, _ in atom_infos:
                    if idx_rgrp == idx_angl + 3:  # 0: backbone / 1: omega / 2: phi
                        cord_dict[atom_name] = tsl_vec + torch.sum(
                            rot_mat * self.cord_dict[resd_name][atom_name].view(1, 3), dim=1)

            # build chiX+1 to chiX transformations
            for idx_angl_src in range(1, n_angls - 1):  # skip the psi angle
                idx_angl_dst = idx_angl_src + 1
                angl_name_src = angl_infos[idx_angl_src][0]
                angl_name_dst = angl_infos[idx_angl_dst][0]
                rot_mat_src, tsl_vec_src = trans_dict[f'{angl_name_src}-bb']
                rot_mat_dst, tsl_vec_dst = trans_dict[f'{angl_name_dst}-bb']
                rot_mat = torch.matmul(rot_mat_src.transpose(1, 0), rot_mat_dst)
                tsl_vec = torch.matmul(rot_mat_src.transpose(1, 0), tsl_vec_dst - tsl_vec_src)
                trans_dict[f'{angl_name_dst}-{angl_name_src}'] = (rot_mat, tsl_vec)

            # record the transformation dict for the current residue type
            trans_dict_full[resd_name] = trans_dict

        return trans_dict_full

    @classmethod
    def __fa2cord_ca(cls, fram_tns_bb, fmsk_mat_bb):
        """Convert backbone local frames to CA-atom 3D coordinates."""

        # initialization
        device = fram_tns_bb.device
        n_resds = fram_tns_bb.shape[0]

        # convert backbone local frames to CA-atom 3D coordinates
        idx_atom_ca = 1  # CA atom is always the 2nd atom, regardless of the residue type
        cord_tns = torch.zeros((n_resds, constants.N_ATOMS_PER_RESD, 3), dtype=torch.float32, device=device)
        cmsk_mat = torch.zeros((n_resds, constants.N_ATOMS_PER_RESD), dtype=torch.int8, device=device)
        cord_tns[:, idx_atom_ca] = fram_tns_bb[:, 0, 3]  # CA: backbone frame's origin point
        cmsk_mat[:, idx_atom_ca] = fmsk_mat_bb[:, 0]

        return cord_tns, cmsk_mat

    def __fa2cord_fa(self, resd_name, fram_tns_bb, fmsk_mat_bb, angl_tns,
                     amsk_mat):  # pylint: disable=too-many-arguments,too-many-locals
        """Convert backbone local frames & torsion angles to full-atom 3D coordinates.

        Args:
        * resd_name: residue name
        * fram_tns_bb: backbone local frames of size L' x 1 x 4 x 3
        * fmsk_mat_bb: backbone local frames' validness masks of size L' x 1
        * angl_tns: torsion angles of size L' x K x 2
        * amsk_mat: torsion angles' validness masks of size L' x K

        Returns:
        * cord_tns: 3D coordinates of size L' x M x 3
        * cmsk_mat: 3D coordinates' validness masks of size L' x M
        """

        # initialization
        device = fram_tns_bb.device
        n_resds = fram_tns_bb.shape[0]
        atom_names_all = constants.ATOM_NAMES_PER_RESD[resd_name]
        atom_names_pad = atom_names_all + ['X'] * (constants.N_ATOMS_PER_RESD - len(atom_names_all))
        atom_infos = constants.ATOM_INFOS_PER_RESD[resd_name]
        angl_infos = constants.ANGL_INFOS_PER_RESD[resd_name]

        # initialize the dict of 3D coordinates
        cord_mat_dict = defaultdict(
            lambda: torch.zeros((n_resds, 3), dtype=torch.float32, device=device))
        cmsk_vec_dict = defaultdict(lambda: torch.zeros((n_resds), dtype=torch.int8, device=device))

        # initialize the dict of rigid-body transformations & validness masks
        trans_dict = {'bb': (fram_tns_bb[:, 0, :3], fram_tns_bb[:, 0, 3])}  # backbone frame
        fmsk_vec_dict = {'bb': fmsk_mat_bb[:, 0]}

        # determine 3D coordinates of atoms belonging to the backbone rigid-group
        rot_tns_curr, tsl_mat_curr = trans_dict['bb']
        atom_names = [x[0] for x in atom_infos if x[1] == 0]
        for atom_name in atom_names:
            cord_vec = self.cord_dict[resd_name][atom_name].to(device)
            cord_mat_dict[atom_name] = \
                tsl_mat_curr + torch.sum(rot_tns_curr * cord_vec.view(1, 1, 3), dim=2)
            cmsk_vec_dict[atom_name] = fmsk_mat_bb[:, 0]

        # determine 3D coordinates of atoms belonging to side-chain rigid-groups
        for idx_angl, (angl_name, _, _) in enumerate(angl_infos):
            # determine current & previous rigid-groups
            rgrp_name_curr = angl_name  # rigid-groups are named after torsion angles
            if rgrp_name_curr in ['psi', 'chi1']:
                rgrp_name_prev = 'bb'
            else:
                cgrp_prev = int(rgrp_name_curr[-1]) - 1
                rgrp_name_prev = f'chi{cgrp_prev}'

            # record the current rigid-group's validness masks
            fmsk_vec_prev = fmsk_vec_dict[rgrp_name_prev]
            fmsk_vec_curr = fmsk_vec_prev * amsk_mat[:, idx_angl + 2]  # skip omega & phi
            fmsk_vec_dict[rgrp_name_curr] = fmsk_vec_curr

            # obtain the relative transformation w.r.t. the previous rigid-body
            rot_tns_prev, tsl_mat_prev = trans_dict[rgrp_name_prev]
            rot_mat_base, tsl_vec_base = \
                self.trans_dict_base[resd_name][f'{rgrp_name_curr}-{rgrp_name_prev}']
            rot_tns_base = rot_mat_base.unsqueeze(dim=0).to(device)
            tsl_mat_base = tsl_vec_base.unsqueeze(dim=0).to(device)
            rot_tns_addi, tsl_mat_addi = self.__build_trans_from_angl(angl_tns[:, idx_angl + 2])
            rot_tns_curr, tsl_mat_curr = self.__combine_trans(
                rot_tns_prev, tsl_mat_prev, rot_tns_base, tsl_mat_base, rot_tns_addi, tsl_mat_addi)
            trans_dict[rgrp_name_curr] = (rot_tns_curr, tsl_mat_curr)

            # map idealized 3D coordinates to the current rigid-group frame
            atom_names = [x[0] for x in atom_infos if x[1] == idx_angl + 3]
            for atom_name in atom_names:
                cord_vec = self.cord_dict[resd_name][atom_name].to(device)
                cord_mat_dict[atom_name] = \
                    tsl_mat_curr + torch.sum(rot_tns_curr * cord_vec.view(1, 1, 3), dim=2)
                cmsk_vec_dict[atom_name] = fmsk_vec_curr

        # packing 3D coordinates & side-chain local frames into tensors
        cord_tns = torch.stack([cord_mat_dict[x] for x in atom_names_pad], dim=1)
        cmsk_mat = torch.stack([cmsk_vec_dict[x] for x in atom_names_pad], dim=1)

        return cord_tns, cmsk_mat

    @classmethod
    def __get_dihd_atoms(cls, aa_seq, cord_tns, cmsk_mat):  # pylint: disable=too-many-locals
        """Get atoms defining all the dihedral angles.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: full-atom 3D coordinates of size L x M x 3
        * cmsk_mat: full-atom 3D coordinates' validness masks of size L x M

        Returns:
        * cord_tns_dihd: dihedral-angle-atom 3D coordinates of size L x K x 4 x 3
        * cmsk_tns_dihd: dihedral-angle-atom 3D coordinates' validness masks of size L x K x 4
        """

        # initialization
        n_resds = len(aa_seq)
        device = cord_tns.device

        # get CA & C atoms' 3D coordinates (with 1-residue offset) to compute omega & phi angles
        cord_mat_ca = ProtStruct.get_atoms(aa_seq, cord_tns, ['CA'])
        cmsk_vec_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])
        cord_mat_c = ProtStruct.get_atoms(aa_seq, cord_tns, ['C'])
        cmsk_vec_c = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['C'])
        cord_mat_cap = torch.cat([torch.zeros_like(cord_mat_ca[:1]), cord_mat_ca[:-1]], dim=0)
        cmsk_vec_cap = torch.cat([torch.zeros_like(cmsk_vec_ca[:1]), cmsk_vec_ca[:-1]], dim=0)
        cord_mat_cp = torch.cat([torch.zeros_like(cord_mat_c[:1]), cord_mat_c[:-1]], dim=0)
        cmsk_vec_cp = torch.cat([torch.zeros_like(cmsk_vec_c[:1]), cmsk_vec_c[:-1]], dim=0)

        # determine atom indices for all the diheral angles
        amsk_vec_dict = {}  # dict of validness masks, each of size K
        idxs_vec_dict = {}  # dict of atom indices, each of size K x 4
        for resd_name in constants.RESD_NAMES_3C:
            # initialization
            atom_names_all = constants.ATOM_NAMES_PER_RESD[resd_name]
            amsk_vec = np.zeros((constants.N_ANGLS_PER_RESD), dtype=np.int8)
            idxs_mat = np.zeros((constants.N_ANGLS_PER_RESD, 4), dtype=np.int64)

            # omega (CA_p - C_p - N - CA)
            amsk_vec[0] = 1  # always valid, except for the 1st residue
            idxs_mat[0, 2] = atom_names_all.index('N')
            idxs_mat[0, 3] = atom_names_all.index('CA')

            # phi (C_p - N - CA - C)
            amsk_vec[1] = 1  # always valid, except for the 1st residue
            idxs_mat[1, 1] = atom_names_all.index('N')
            idxs_mat[1, 2] = atom_names_all.index('CA')
            idxs_mat[1, 3] = atom_names_all.index('C')

            # psi, chi1, chi2, chi3, and chi4
            for idx_rgrp, (_, _, atom_names_sel) in enumerate(constants.ANGL_INFOS_PER_RESD[resd_name]):
                amsk_vec[idx_rgrp + 2] = 1
                for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                    idxs_mat[idx_rgrp + 2, idx_atom_sel] = atom_names_all.index(atom_name_sel)

            # record atom indices for the current residue type
            amsk_vec_dict[resd_name] = amsk_vec
            idxs_vec_dict[resd_name] = idxs_mat.ravel()

        # expand atom indices based on residue types
        amsk_vec_list = []
        idxs_vec_list = []
        for resd_name_1c in aa_seq:
            resd_name_3c = constants.RESD_MAP_1TO3[resd_name_1c]
            amsk_vec_list.append(amsk_vec_dict[resd_name_3c])
            idxs_vec_list.append(idxs_vec_dict[resd_name_3c])
        amsk_mat_full = torch.tensor(np.stack(amsk_vec_list), dtype=torch.int8, device=device)
        amsk_mat_full[0, :2] = 0  # 1st residue's omega & phi angles are invalid
        idxs_mat_full = torch.tensor(np.stack(idxs_vec_list), dtype=torch.int64, device=device)

        # extract 3D coordinates for torsion angle computation
        cord_tns_dihd = torch.gather(
            cord_tns, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, 3),
        ).view(n_resds, constants.N_ANGLS_PER_RESD, 4, 3)
        cord_tns_dihd[:, 0, 0] = cord_mat_cap  # omega - CA_prev
        cord_tns_dihd[:, 0, 1] = cord_mat_cp  # omega - C_prev
        cord_tns_dihd[:, 1, 0] = cord_mat_cp  # phi - C_prev

        # extract validness masks for torsion angle computation
        cmsk_tns_dihd = torch.gather(cmsk_mat, 1, idxs_mat_full).view(n_resds, constants.N_ANGLS_PER_RESD, 4)
        cmsk_tns_dihd[:, 0, 0] = cmsk_vec_cap  # omega - CA_prev
        cmsk_tns_dihd[:, 0, 1] = cmsk_vec_cp  # omega - C_prev
        cmsk_tns_dihd[:, 1, 0] = cmsk_vec_cp  # omega - C_prev
        cmsk_tns_dihd *= amsk_mat_full.unsqueeze(dim=2)

        return cord_tns_dihd, cmsk_tns_dihd

    @classmethod
    def __combine_trans(cls, rot_tns_1, tsl_mat_1, rot_tns_2, tsl_mat_2, *args):
        """Combine two or more transformations."""

        # combine the first two transformations
        # rot_tns = torch.bmm(rot_tns_1, rot_tns_2)  # much slower!
        rot_tns = torch.sum(rot_tns_1.unsqueeze(dim=3) * rot_tns_2.unsqueeze(dim=1), dim=2)
        tsl_mat = torch.sum(rot_tns_1 * tsl_mat_2.unsqueeze(dim=1), dim=2) + tsl_mat_1

        # recursively process remaining transformations
        if len(args) > 0:
            assert len(args) % 2 == 0, \
                'rotation matrices and translation vectors must be provided simultaneously'
            return cls.__combine_trans(rot_tns, tsl_mat, *args)

        return rot_tns, tsl_mat

    @classmethod
    def __build_trans_from_angl(cls, angl_mat):
        """Build rigid-body transformations from angles (represented as cosine & sine values)."""

        # initialization
        device = angl_mat.device
        n_resds = angl_mat.shape[0]

        # build rigid-body transformations
        one_vec = torch.ones((n_resds), dtype=torch.float32, device=device)
        zro_vec = torch.zeros((n_resds), dtype=torch.float32, device=device)
        cos_vec = angl_mat[:, 0]
        sin_vec = angl_mat[:, 1]
        rot_tns = torch.stack([
            torch.stack([one_vec, zro_vec, zro_vec], dim=1),
            torch.stack([zro_vec, cos_vec, -sin_vec], dim=1),
            torch.stack([zro_vec, sin_vec, cos_vec], dim=1),
        ], dim=1)
        tsl_mat = torch.zeros((n_resds, 3), dtype=torch.float32, device=device)

        return rot_tns, tsl_mat

    @classmethod
    def atom37_to_atom14(self, aa_seq, cord37_tns, cmsk37_mat):  # pylint: disable=too-many-locals
        """Convert 3D coordinates with atom37 to 3D coordinates with atom14.

        Args:
            aa_seq: amino-acid sequence
            cord37_tns: 3D coordinates of size L x 37 x 3
            cmsk37_mat: 3D coordinates' validness masks of size L x 37

        Returns:
            cord14_tns: 3D coordinates of size L x 14 x 3
            cmsk14_mat: 3D coordinates' validness masks of size L x 14
        """

        np_aatype = np.array([constants.RESD_ORDER_WITH_X[resd] for resd in list(aa_seq)])

        per_res_idx = constants.restype_atom14_to_atom37[np_aatype]
        res_idx = np.tile(np.arange(per_res_idx.shape[0])[..., None], (1, per_res_idx.shape[1]))

        atom14_pos_mask = constants.restype_atom14_mask[np_aatype]
        atom14_pos_mask = torch.from_numpy(atom14_pos_mask).type_as(cmsk37_mat)
        atom14_mask = cmsk37_mat[res_idx, per_res_idx]
        atom14_mask = atom14_mask * atom14_pos_mask

        cord14_tns = cord37_tns[res_idx, per_res_idx]
        cord14_tns = cord14_tns * atom14_mask[..., None]

        atom14_mask = atom14_mask.reshape(len(aa_seq), 14)
        cord14_tns = cord14_tns.reshape(len(aa_seq), 14, 3)

        return cord14_tns, atom14_mask

    @classmethod
    def atom14_to_atom37(self, aa_seq, cord14_tns, cmsk14_mat):
        """Convert 3D coordinates with atom14 to 3D coordinates with atom37.

        Args:
            aa_seq: amino-acid sequence
            cord14_tns: 3D coordinates of size L x 14 x 3
            cmsk14_mat: 3D coordinates' validness masks of size L x 14

        Returns:
            cord37_tns: 3D coordinates of size L x 37 x 3
            cmsk37_mat: 3D coordinates' validness masks of size L x 37
        """
        np_aatype = np.array([constants.RESD_ORDER_WITH_X[resd] for resd in list(aa_seq)])

        per_res_idx = constants.restype_atom37_to_atom14[np_aatype]
        res_idx = np.tile(np.arange(per_res_idx.shape[0])[..., None], (1, per_res_idx.shape[1]))

        atom37_pos_mask = constants.restype_atom37_mask[np_aatype]
        atom37_pos_mask = torch.from_numpy(atom37_pos_mask).type_as(cmsk14_mat)
        cmsk37_mat = cmsk14_mat[res_idx, per_res_idx]
        cmsk37_mat = cmsk37_mat * atom37_pos_mask

        cord37_tns = cord14_tns[res_idx, per_res_idx]
        cord37_tns = cord37_tns * cmsk37_mat[..., None]

        cmsk37_mat = cmsk37_mat.reshape(len(aa_seq), 37)
        cord37_tns = cord37_tns.reshape(len(aa_seq), 37, 3)

        return cord37_tns, cmsk37_mat
