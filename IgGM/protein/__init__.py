# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
from .data_transform import calc_ppi_sites
from .parser import PdbParser, parse_a3m, parse_fasta, export_fasta
from .atom_mapper import AtomMapper
from .prot_struct import ProtStruct
from .prot_converter import ProtConverter
from .pdb_fixer import PdbFixer

def cal_ppi(pdb_path, complex_ids):
    """Calculate PPI sites"""
    prot_data = {}
    ligand_id = "HL"
    aa_seqs = ""
    atom_cords = []
    atom_cmsks = []
    for chn_id in complex_ids:
        if chn_id == "A":
            continue
        aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=chn_id)
        aa_seqs += aa_seq
        atom_cords.append(atom_cord)
        atom_cmsks.append(atom_cmsk)

    prot_data[ligand_id] = {"seq": aa_seqs, "cord": torch.cat(atom_cords, dim=0), "cmsk": torch.cat(atom_cmsks, dim=0)}

    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id="A")
    prot_data["A"] = {"seq": aa_seq, "cord": atom_cord, "cmsk": atom_cmsk}
    ppi_data = calc_ppi_sites(prot_data, ["A", ligand_id])
    return ppi_data['A']
