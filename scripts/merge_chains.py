# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys
from collections import OrderedDict

# Finds the root directory (one level up from this script's folder)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import torch
import tqdm

from IgGM.protein import export_fasta
from IgGM.protein.parser import parse_fasta, PdbParser


def parse_args():
    parser = argparse.ArgumentParser(description='Merge antigen chains')
    parser.add_argument('--antigen', '-ag', type=str, required=True,
                        help='Directory path to input antigen PDB files')
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directory path to output PDB files, default is "outputs"',
    )
    parser.add_argument(
        '--antibody_ids',
        default="H_L",
        type=str,
        help='antibody chain ids',
    )
    parser.add_argument(
        '--merge_ids',
        default="A",
        type=str,
        help='merge chain ids',
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    antibody_ids = args.antibody_ids.split('_')
    merge_ids = args.merge_ids.split('_')
    pdb_file = args.antigen

    sequences = []
    ids = []
    chains = OrderedDict()

    for chn_id in antibody_ids:
        aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_file, chain_id=chn_id)
        if aa_seq is None:
            continue
        chains[chn_id] = {
            'seq': aa_seq,
            'cord': atom_cord,
            'cmsk': atom_cmsk,
        }
        sequences.append(aa_seq)
        ids.append(chn_id)

    antigen_seq = ""
    antigen_cord = []
    antigen_cmsk = []
    for chn_id in merge_ids:
        aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_file, chain_id=chn_id)
        if aa_seq is None:
            continue
        antigen_seq += aa_seq
        antigen_cord.append(atom_cord)
        antigen_cmsk.append(atom_cmsk)

    chains['A'] = {
        'seq': antigen_seq,
        'cord': torch.cat(antigen_cord, dim=0),
        'cmsk': torch.cat(antigen_cmsk, dim=0),
    }
    sequences.append(antigen_seq)
    ids.append('A')

    pdb_name = os.path.basename(pdb_file)
    fasta_output = os.path.join(args.output, pdb_name[:-4] + '_merge.fasta')
    pdb_output = os.path.join(args.output, pdb_name[:-4] + '_merge.pdb')
    export_fasta(sequences, ids=ids, output=fasta_output)
    PdbParser.save_multimer(chains, pdb_output)
    print(f'Merged chains saved to {pdb_output}')


if __name__ == '__main__':
    main()
