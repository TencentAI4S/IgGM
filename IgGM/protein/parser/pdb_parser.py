# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 17:08
import gzip
import logging
import os
import warnings

import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

from IgGM.utils import get_rand_str
from .fasta_parser import parse_fasta
from ..prot_constants import RESD_NAMES_3C, RESD_MAP_1TO3, RESD_MAP_3TO1, ATOM_NAMES_PER_RESD, N_ATOMS_PER_RESD


class PdbParseError(Exception):
    """Exceptions raised when parsing a PDB file w/ BioPython."""


class PdbParser:
    """Parser for PDB files."""

    @classmethod
    def load(
            cls, pdb_fpath, aa_seq=None, fas_fpath=None,
            model_id=None, chain_id=None, has_plddt=False,
    ):
        """Load a protein structure from the PDB file.

        Args:
        * pdb_fpath: path to the PDB file
        * aa_seq: (optional) reference amino-acid sequence
        * fas_fpath: (optional) path to the reference FASTA file
        * model_id: (optional) model ID
        * chain_id: (optional) chain ID
        * has_plddt: (optional) whether the PDB file contains per-residue & full-chain pLDDT scores

        Returns:
        * aa_seq: amino-acid sequence
        * atom_cords: per-atom 3D coordinates of size L x M x 3
        * atom_masks: per-atom 3D coordinates' validness masks of size L x M
        * meta_data: dict of meta-data stored in the PDB file
        * error_msg: error message raised when parsing the PDB file

        Note:
        * The GZ-compressed PDB file can be provided with a suffix of ".gz".
        * The amino-acid sequence is determined in the following order:
          a) parsed from the FASTA file
          b) parsed from SEQRES records in the PDB file
          c) parsed from ATOM records in the PDB file
        * If <chain_id> is not provided, then the first chain will be returned. The specific order
          is defined by the <BioPython> package. If <chain_id> is provided, then the first model
          with the specified chain ID will be returned.
        """

        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        if fas_fpath is not None:
            logging.debug('FASTA file provided: %s', fas_fpath)

        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParseError('PDB_FILE_NOT_FOUND')
            if not (pdb_fpath.endswith('.pdb') or pdb_fpath.endswith('.gz')):
                raise PdbParseError('PDB_FILE_FORMAT_NOT_SUPPORTED')
            if (fas_fpath is not None) and (not os.path.exists(fas_fpath)):
                raise PdbParseError('FASTA_FILE_NOT_FOUND')

            # obtain the amino-acid sequence (could be None)
            if aa_seq is None:
                if fas_fpath is not None:
                    aa_seq = parse_fasta(fas_fpath)[0][0]
                else:  # then the amino-acid sequence must be parsed from the PDB file
                    aa_seq = cls.__get_aa_seq_from_seqres(pdb_fpath, chain_id)

            # parse the PDB file w/ biopython
            structure = cls.__get_structure(pdb_fpath)

            # obtain meta-data from the structure
            meta_data = {
                'id': structure.header['idcode'],
                'date': structure.header['release_date'],
                'reso': structure.header['resolution'],
                'mthd': structure.header['structure_method'],
            }

            # find the first chain matching the model/chain ID
            chain = cls.__get_chain(structure, model_id, chain_id)

            # obtain atom coordinates & validness masks
            aa_seq, atom_cords, atom_masks = cls.__get_atoms(chain, aa_seq)

            # obtain pLDDT scores (per-residue & full-chain)
            if has_plddt:
                meta_data['plddt-r'], meta_data['plddt-c'] = \
                    cls.__get_plddt(pdb_fpath, aa_seq, chain_id)

            # set the error message to None
            error_msg = None
        except PdbParseError as error:
            aa_seq, atom_cords, atom_masks, meta_data, error_msg = None, None, None, None, error

        return aa_seq, atom_cords, atom_masks, meta_data, error_msg

    @classmethod
    def save(cls, aa_seq, atom_cords, atom_masks, path, chain_id='A',
             plddt_vec=None):  # pylint: disable=too-many-arguments,too-many-locals
        """Save the protein structure to a PDB file.

        Args:
            aa_seq: amino-acid sequence
            atom_cords: per-atom 3D coordinates of size L x M x 3
            atom_masks: per-atom 3D coordinates' validness masks of size L x M
            path: path to the PDB file
            chain_id: (optional) chain ID
            plddt_vec: (optional) per-residue lDDT-Ca predicted scores of size L
        """

        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w', encoding='UTF-8') as o_file:
            # record the predicted lDDT-Ca score
            if plddt_vec is not None:
                plddt_val = torch.mean(plddt_vec).item()  # full-chain lDDT-Ca predicted score
                o_file.write('REMARK 250\n')
                o_file.write(f'REMARK 250 Predicted lDDT-Ca score: {plddt_val:.4f}\n')

            # record per-atom 3D coordinates
            pdb_strs = cls.__get_pdb_strs(
                aa_seq, chain_id, atom_cords, atom_masks, plddt_vec=plddt_vec)
            o_file.write('\n'.join(pdb_strs) + '\n')

    @classmethod
    def save_multimer(cls, prot_data, path, pred_info=''):
        """Save the multimer structure to a PDB file (also works for single-chain inputs).

        Args:
            prot_data: (ordered) dict of multimer structure
            path: path to the PDB file
            pred_info: predicted information

        Notes:
        * <prot_data> uses <chain_id> as key, and the value dict has following fields:
          > seq / cord / cmsk / plddt (optional)
        """

        # initialization
        chain_ids = sorted(list(prot_data.keys()))
        has_plddt = ('plddt' in prot_data[chain_ids[0]])
        has_bfctr = ('bfctr' in prot_data[chain_ids[0]])

        # export the multimer structure to a PDB file
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
        with open(path, 'w', encoding='UTF-8') as o_file:
            # record the overall pLDDT score
            if has_plddt or (pred_info != ''):
                o_file.write('REMARK 250\n')
            if has_plddt:
                plddt_vec = torch.cat([x['plddt'] for x in prot_data.values()], dim=0)
                plddt_val = torch.mean(plddt_vec).item()
                o_file.write(f'REMARK 250 Predicted lDDT-Ca score: {plddt_val:.4f}\n')
            if pred_info != '':
                o_file.write(pred_info)

            # enumerate over all the chains in the multimer
            idx_atom_base = 0
            for chain_id, chain_data in prot_data.items():
                pdb_strs = cls.__get_pdb_strs(
                    chain_data['seq'], chain_id, chain_data['cord'], chain_data['cmsk'],
                    plddt_vec=(chain_data['plddt'] if has_plddt else None),
                    bfctr_vec=(chain_data['bfctr'] if has_bfctr else None),
                    idx_atom_base=idx_atom_base,
                )
                o_file.write('\n'.join(pdb_strs) + '\n')
                idx_atom_base += torch.sum(chain_data['cmsk']).item()

    @classmethod
    def __get_pdb_strs(
            cls, aa_seq, chain_id, atom_cords, atom_masks,
            plddt_vec=None, bfctr_vec=None, idx_atom_base=0,
    ):
        """Get PDB line strings (no REMARK section) for the specified protein structure."""
        occupancy = 1.0
        temp_fctr = 1.0
        cord_min = -999.999
        cord_max = 9999.999

        # reset invalid values in atom coordinates
        atom_cords_vald = torch.clip(atom_cords, cord_min, cord_max)
        atom_cords_vald[torch.isnan(atom_cords_vald)] = 0.0
        atom_cords_vald[torch.isinf(atom_cords_vald)] = 0.0

        # get PDB line strings for ATOM records
        n_atoms = 0
        pdb_strs = []
        idx_resd_vld = -1  # last valid residue index
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            if torch.sum(atom_masks[idx_resd]) == 0:
                continue
            idx_resd_vld = idx_resd
            if plddt_vec is not None:
                temp_fctr = plddt_vec[idx_resd]  # replace the temperature factor
            elif bfctr_vec is not None:
                temp_fctr = bfctr_vec[idx_resd]
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            atom_names = ATOM_NAMES_PER_RESD[resd_name_3c]
            for idx_atom, atom_name in enumerate(atom_names):
                if atom_masks[idx_resd, idx_atom] == 0:
                    continue
                n_atoms += 1
                line_str = ''.join([
                    'ATOM  ',
                    f'{n_atoms + idx_atom_base:5d}',
                    f'  {atom_name:3}',
                    ' ',  # alternative location indicator
                    resd_name_3c,
                    f' {chain_id}',
                    f'{idx_resd + 1:4d}',
                    ' ' * 4,  # code for insertion of residues
                    f'{atom_cords_vald[idx_resd, idx_atom, 0]:8.3f}',  # coordinate - X
                    f'{atom_cords_vald[idx_resd, idx_atom, 1]:8.3f}',  # coordinate - Y
                    f'{atom_cords_vald[idx_resd, idx_atom, 2]:8.3f}',  # coordinate - Z
                    f'{occupancy:6.2f}',
                    f'{temp_fctr:6.2f}',
                    ' ' * 10,  # empty
                    f' {atom_name[0]}',
                    ' ' * 2,  # charge on the atom
                ])
                pdb_strs.append(line_str)

        # append the TER record
        if idx_resd_vld != -1:
            n_atoms += 1
            line_str = ''.join([
                'TER   ',
                f'{n_atoms + idx_atom_base:5}',
                ' ' * 6,
                resd_name_3c,
                f' {chain_id}',
                f'{idx_resd_vld + 1:4}',
                ' ',  # insertion code
            ])
            pdb_strs.append(line_str)

        return pdb_strs

    @classmethod
    def get_chain_ids(cls, path):
        """Get a list of chain IDs.

        Args:
        * path: path to the PDB file

        Returns:
        * chain_ids: list of chain IDs
        """

        # obtain line strings from the PDB file
        line_strs = cls.__get_line_strs(path)

        # get a list of chain IDs
        chain_ids = set()
        for line_str in line_strs:
            if line_str.startswith('ATOM'):
                chain_ids.add(line_str[21])
        chain_ids = sorted(list(chain_ids))

        return chain_ids

    @classmethod
    def __get_aa_seq_from_seqres(cls, path, chain_id):
        """Get the amino-acid sequence from SEQRES records."""

        # get residue names from SEQRES records
        resd_names = []
        line_strs = cls.__get_line_strs(path)
        for line_str in line_strs:
            if not line_str.startswith('SEQRES'):
                continue
            if (chain_id is not None) and (line_str[11] != chain_id):
                continue
            resd_names.extend(line_str[19:].split())

        # convert residue names into the amino-acid sequence
        if len(resd_names) == 0:  # no SEQRES records found
            aa_seq = None
        else:
            for resd_name in resd_names:
                if resd_name not in RESD_NAMES_3C:
                    raise PdbParseError('HAS_UNKNOWN_RESIDUE')
            aa_seq = ''.join([RESD_MAP_3TO1[x] for x in resd_names])

        return aa_seq

    @classmethod
    def __get_structure(cls, path):
        """Get the structure from the PDB file."""

        try:
            parser = PDBParser()
            if path.endswith('.pdb'):
                with open(path, 'r', encoding='UTF-8') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
            else:  # then <path> must end with '.gz'
                with gzip.open(path, 'rt') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParseError('BIOPYTHON_FAILED_TO_PARSE') from error

        return structure

    @classmethod
    def __get_chain(cls, structure, model_id, chain_id):
        """Get the first chain matching the specified chain ID (could be None)."""

        chain = None
        for model in structure:
            if (model_id is not None) and (model.get_id() != model_id):
                continue
            for chain_curr in model:
                if (chain_id is None) or (chain_curr.get_id() == chain_id):
                    chain = chain_curr
                    break
            if chain is not None:
                break

        # check whether the specified chain has been found
        if chain is None:
            raise PdbParseError('CHAIN_NOT_FOUND')

        return chain

    @classmethod
    def __get_atoms(cls, chain, aa_seq):  # pylint: disable=too-many-locals
        """Get atom coordinates & masks from the specified chain."""

        # get discontinous segments and align them to the full amino-acid sequence
        seg_infos = cls.__get_segs(chain)
        if aa_seq is None:
            aa_seq = cls.__build_seq_from_segs(seg_infos)
        cls.__align_segs_to_seq(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        idx_resd_prev = -9999
        ins_code_prev = ' '  # no insertion
        n_resds_ins = 0  # number of inserted residues
        atom_cords = torch.zeros((seq_len, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
        atom_masks = torch.zeros((seq_len, N_ATOMS_PER_RESD), dtype=torch.int8)
        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if (idx_resd == idx_resd_prev) and (ins_code != ins_code_prev):
                n_resds_ins += 1
            idx_resd_prev = idx_resd
            ins_code_prev = ins_code

            # determine the offset for the current segment
            seg_infos_sel = [x for x in seg_infos if x['ib'] <= idx_resd + n_resds_ins < x['ie']]
            if len(seg_infos_sel) != 1:
                raise PdbParseError('MULTIPLE_MATCHED_SEGMENTS')
            offset = seg_infos_sel[0]['offset']

            # update atom coordinates & masks
            atom_names = ATOM_NAMES_PER_RESD[residue.get_resname()]
            for idx_atom, atom_name in enumerate(atom_names):
                if residue.has_id(atom_name):
                    atom_cords[idx_resd + n_resds_ins + offset, idx_atom] = \
                        torch.from_numpy(residue[atom_name].get_coord())
                    atom_masks[idx_resd + n_resds_ins + offset, idx_atom] = 1

        return aa_seq, atom_cords, atom_masks

    @classmethod
    def __get_plddt(cls, path, aa_seq, chain_id):
        """Get pLDDT scores (per-residue & overall).

        Notes:
            The overall pLDDT is computed all chains in the PDB file, rather than a single chain.
        """
        n_resds = len(aa_seq)
        # obtain line strings from the PDB file
        line_strs = cls.__get_line_strs(path)

        # get pLDDT scores (per-residue & overall)
        plddt_val_full = None
        plddt_vec_resd = torch.zeros((n_resds), dtype=torch.float32)
        for line_str in line_strs:
            if line_str.startswith('REMARK 250 Predicted lDDT-Ca score:'):
                plddt_val_full = torch.tensor([float(line_str.split()[-1])], dtype=torch.float32)
                continue
            if not line_str.startswith('ATOM'):
                continue
            if (chain_id is not None) and (line_str[21] != chain_id):
                continue
            if line_str[12:16].strip() == 'CA':
                idx_resd = int(line_str[22:26]) - 1
                if (idx_resd < 0) or (idx_resd >= n_resds):
                    raise PdbParseError('INVALID_RESIDUE_INDEX')
                plddt_vec_resd[idx_resd] = float(line_str[60:66])
        if plddt_val_full is None:
            plddt_val_full = torch.mean(plddt_vec_resd).reshape(-1)

        # re-scale pLDDT scores to [0, 1]
        if plddt_val_full.item() > 1.0:
            plddt_val_full /= 100.0
            plddt_vec_resd /= 100.0

        return plddt_vec_resd, plddt_val_full

    @classmethod
    def __get_segs(cls, chain):
        """Get discontinous segments of amino-acid sequences."""

        seg_infos = []
        idx_resd_prev = -9999
        ins_code_prev = ' '  # no insertion
        n_resds_ins = 0  # number of inserted residues
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if resd_name not in RESD_NAMES_3C:
                raise PdbParseError('HAS_UNKNOWN_RESIDUE')
            if (idx_resd == idx_resd_prev) and (ins_code != ins_code_prev):
                n_resds_ins += 1
            idx_resd_prev = idx_resd
            ins_code_prev = ins_code

            # update the last segment, or add a new segment
            if (len(seg_infos) >= 1) and (seg_infos[-1]['ie'] == idx_resd + n_resds_ins):
                seg_infos[-1]['ie'] += 1
                seg_infos[-1]['seq'] += RESD_MAP_3TO1[resd_name]
            else:
                seg_infos.append({
                    'ib': idx_resd + n_resds_ins,  # inclusive
                    'ie': idx_resd + n_resds_ins + 1,  # exclusive
                    'seq': RESD_MAP_3TO1[resd_name],
                })

        return seg_infos

    @classmethod
    def __build_seq_from_segs(cls, seg_infos):
        """Build the full amino-acid sequence from discontinous segments."""

        seg_infos.sort(key=lambda x: x['ib'])
        aa_seq_list = []
        for idx_seg, seg_info in enumerate(seg_infos):
            if idx_seg != 0:
                gap = seg_info['ib'] - seg_infos[idx_seg - 1]['ie']
                aa_seq_list.append('X' * gap)
            aa_seq_list.append(seg_info['seq'])
        aa_seq = ''.join(aa_seq_list)

        return aa_seq

    @classmethod
    def __align_segs_to_seq(cls, seg_infos, aa_seq):
        """Align discontinous segments to the full amino-acid sequence."""

        mask_vec = torch.zeros(len(aa_seq), dtype=torch.int8)
        is_valid = cls.__align_segs_to_seq_impl(seg_infos, aa_seq, 0, mask_vec)
        if not is_valid:
            raise PdbParseError('NO_VALID_OFFSET')

    @classmethod
    def __align_segs_to_seq_impl(cls, seg_infos, aa_seq, idx_seg, mask_vec):
        """Align discontinous segments to the full amino-acid sequence - core implementation."""

        # find all the matching sub-strings w/ overlapping segments allowed
        # NOTE we do not use re.finditer() here since it does not allow overlapping segments
        def _find_all_matches(seq_base, seq_qury):
            idx_base = 0
            idxs_beg = []  # list of starting indices
            while seq_qury in seq_base[idx_base:]:
                idx_beg = idx_base + seq_base[idx_base:].index(seq_qury)
                idx_base = idx_beg + 1
                idxs_beg.append(idx_beg)
            return idxs_beg

        if idx_seg == len(seg_infos):
            return True

        seg_info = seg_infos[idx_seg]
        seg_len = len(seg_info['seq'])
        idxs_resd_beg = _find_all_matches(aa_seq, seg_info['seq'])
        for idx_resd_beg in idxs_resd_beg:
            idx_resd_end = idx_resd_beg + seg_len
            if torch.max(mask_vec[idx_resd_beg:]) == 0:  # do not allow backward alignment
                offset = idx_resd_beg - seg_info['ib']
                mask_vec[idx_resd_beg:idx_resd_end] = 1  # mark as occupied
                is_valid = cls.__align_segs_to_seq_impl(seg_infos, aa_seq, idx_seg + 1, mask_vec)
                if is_valid:
                    seg_info['offset'] = offset
                    return True
                mask_vec[idx_resd_beg:idx_resd_end] = 0  # mark as unoccupied

        return False

    @classmethod
    def __get_line_strs(cls, path):
        """Get line strings from the PDB file."""

        # obtain line strings from the PDB file
        if path.endswith('.pdb'):
            with open(path, 'r', encoding='UTF-8') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]
        else:  # then <path> must end with '.gz'
            with gzip.open(path, 'rt') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]

        return i_lines
