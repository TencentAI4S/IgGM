# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from .data_transform import calc_ppi_sites
from .parser import PdbParser, parse_a3m, parse_fasta, export_fasta
from .atom_mapper import AtomMapper
from .prot_struct import ProtStruct
from .prot_converter import ProtConverter
from .pdb_fixer import PdbFixer


def cal_ppi(pdb_path, complex_ids, sequences):
    """Calculate PPI sites"""
    prot_data = {}
    if len(complex_ids) == 3:
        ligand_id = complex_ids[0] + complex_ids[1]
    else:
        ligand_id = complex_ids[0]
    receptor_id = complex_ids[-1]
    aa_seqs = ""
    atom_cords = []
    atom_cmsks = []
    for chn_id in complex_ids:
        if chn_id == receptor_id:
            continue
        aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=chn_id)
        aa_seqs += aa_seq
        atom_cords.append(atom_cord)
        atom_cmsks.append(atom_cmsk)

    prot_data[ligand_id] = {"seq": aa_seqs, "cord": torch.cat(atom_cords, dim=0), "cmsk": torch.cat(atom_cmsks, dim=0)}

    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=receptor_id, aa_seq=sequences[-1])
    prot_data[receptor_id] = {"seq": aa_seq, "cord": atom_cord, "cmsk": atom_cmsk}
    ppi_data = calc_ppi_sites(prot_data, [receptor_id, ligand_id])
    return ppi_data[receptor_id]


def crop_sequence_with_epitope(sequence, coords, masks, epitope, contact=None, max_len=384):
    if len(sequence) <= max_len:
        return sequence, coords, masks, epitope, contact

    # Find regions with epitope=1
    epitope_indices = torch.where(epitope == 1)[0]
    if len(epitope_indices) == 0:
        # If no epitope regions, take center portion
        start = (len(sequence) - max_len) // 2
        end = start + max_len
    else:
        # Calculate center of epitope region
        epitope_center = epitope_indices.float().mean().int()

        # Calculate start and end to center the window around epitope
        start = max(0, min(
            len(sequence) - max_len,  # Don't exceed sequence length
            epitope_center - max_len // 2  # Center window on epitope
        ))
        end = start + max_len

    # Crop all arrays
    sequence = sequence[start:end]
    coords = coords[start:end]
    masks = masks[start:end]
    epitope = epitope[start:end]

    if contact is not None:
        contact = contact[start:end, start:end]

    return sequence, coords, masks, epitope, contact
