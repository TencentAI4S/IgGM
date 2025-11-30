# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys
from copy import deepcopy

import torch
import tqdm

from IgGM.protein import cal_ppi, crop_sequence_with_epitope

sys.path.append('.')

from IgGM.deploy import AbDesigner
from IgGM.utils import setup
from IgGM.protein.parser import parse_fasta, PdbParser
from IgGM.model.pretrain import esm_ppi_650m_ab, antibody_design_trunk, IGSO3Buffer_trunk


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody sequence and structure co-design w/ IgGM')
    parser.add_argument('--fasta', '-f', type=str, required=True, help='Directory path to input antibody FASTA files, X for design region')
    parser.add_argument('--fasta_origin', '-fo', type=str, required=False, help='Directory path to original antibody FASTA files for affinity maturation')
    parser.add_argument('--antigen', '-ag', type=str, required=True,
                        help='Directory path to input antigen PDB files')
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directory path to output PDB files, default is "outputs"',
    )
    parser.add_argument(
        '--epitope',
        default=None,
        nargs='+', type=int,
        help='epitope residues in antigen chain A , for example: 1 2 3 4 55',
    )
    parser.add_argument(
        '--device', '-d', type=str, default=None, help='inference device'
    )
    parser.add_argument(
        '--steps', '-s', type=int, default=10, help='number of sampling steps'
    )
    parser.add_argument(
        '--chunk_size', '-cs',
        type=int,
        default=64,
        help='chunk size for long chain inference',
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1,
        help='temperature for sampling',
    )
    parser.add_argument(
        '--num_samples', '-ns',
        type=int,
        default=1,
        help='number of samples for each input',
    )
    parser.add_argument(
        '--cal_epitope', '-ce',
        action='store_true',
        default=False,
        help='if use, will calculate epitope from antigen pdb',
    )
    parser.add_argument(
        '--relax', '-r',
        action='store_true',
        help='relax structures after design',
    )
    parser.add_argument(
        '--max_antigen_size', '-mas',
        type=int,
        default=2000,
        help='max size of antigen chain, default is 2000',
    )
    parser.add_argument(
        '--run_task', '-rt',
        type=str,
        default='design',
        choices=['design', 'inverse_design', 'fr_design', 'affinity_maturation'],
        help='design or inverse design, design for antibody sequence and structure design, inverse design for antibody sequence design only',
    )
    args = parser.parse_args()

    return args


def predict(args):
    """Predict antibody & antigen sequence and structures w/ pre-trained IgGM-Ag models."""
    pdb_path = args.antigen
    fasta_path = args.fasta

    sequences, ids, _ = parse_fasta(fasta_path)
    assert len(sequences) in (1, 2, 3), f"must be 1, 2 or 3 chains in fasta file"
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids) if seq_id != ids[-1]]
    _, basename = os.path.split(fasta_path)
    if args.cal_epitope:
        epitope = cal_ppi(pdb_path, ids, sequences)
        epitope = torch.nonzero(epitope).flatten().tolist()
        print(f"epitope: {' '.join(str(i + 1) for i in epitope)}")
        return
    name = basename.split(".")[0]
    output = f"{args.output}/{name}.pdb"

    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id=ids[-1], aa_seq=sequences[-1])
    if args.epitope is None:
        try:
            epitope = cal_ppi(pdb_path, ids, sequences)
        except:
            epitope = args.epitope
    else:
        epitope = torch.zeros(len(aa_seq))
        for i in args.epitope:
            epitope[i - 1] = 1

    if len(aa_seq) > args.max_antigen_size:
        aa_seq, atom_cord, atom_cmsk, epitope, _ = crop_sequence_with_epitope(
            aa_seq, atom_cord, atom_cmsk, epitope, max_len=args.max_antigen_size
        )
    chains.append({"sequence": aa_seq,
                    "cord": atom_cord,
                    "cmsk": atom_cmsk,
                    "epitope": epitope,
                    "id": ids[-1]})

    if args.run_task == 'affinity_maturation':
        batches = []
        replace_sequences, replace_ids, _ = parse_fasta(args.fasta_origin)
        if len(chains) == 3:
            mask_pos = [i for i, char in enumerate(chains[0]['sequence'] + chains[1]['sequence']) if char == 'X']
            if 'X' in replace_sequences[0]:
                replace_seq = chains[0]['sequence'] + replace_sequences[1]
            elif 'X' in replace_sequences[1]:
                replace_seq = replace_sequences[0] + chains[1]['sequence']
            else:
                replace_seq = replace_sequences[0] + replace_sequences[1]
        else:
            mask_pos = [i for i, char in enumerate(chains[0]['sequence']) if char == 'X']
            replace_seq = replace_sequences[0]

        h_seq_len = len(chains[0]['sequence'])
        for i in range(args.num_samples):
            for j in range(len(chains) - 1):
                for pos in mask_pos:
                    new_seq = list(replace_seq)
                    new_seq[pos] = 'X'
                    new_seq = ''.join(new_seq)
                    if j == 0 and pos >= h_seq_len:
                        continue
                    if j == 1 and pos < h_seq_len:
                        continue
                    if len(chains) == 3:
                        chains[0]['sequence'] = new_seq[:h_seq_len]
                        chains[1]['sequence'] = new_seq[h_seq_len:]
                    else:
                        chains[0]['sequence'] = new_seq
                    # print(chains)
                    batches.extend([
                        {
                        "name": replace_ids[j],
                        "chains": deepcopy(chains),
                        "output": f"{args.output}/{replace_ids[j]}_{pos}_{i*args.num_samples + j}.pdb",
                        "replace_chain": replace_seq
                        }
                    ])
    else:
        batches = [
            {
                "name": name,
                "chains": chains,
                "output": f"{args.output}/{name}_{i}.pdb",
            }
            for i in range(args.num_samples)
        ]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # antibody & antigen structures prediction & sequence design
    designer = AbDesigner(
        ppi_path=esm_ppi_650m_ab(),
        design_path=antibody_design_trunk(args.run_task),
        buffer_path=IGSO3Buffer_trunk(),
        config=args,
    )
    designer.to(device)

    chunk_size = args.chunk_size
    temperature = args.temperature
    print(f"#inference samples: {len(batches)}")

    # for multiple runs
    import time
    import random
    random.seed(time.time())
    random.shuffle(batches)

    for task in tqdm.tqdm(batches):
        if os.path.exists(task["output"]):
            print(f'{task["output"]} exists or has been executed by other process')
            continue
        designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size, relax=args.relax,  temperature=temperature, task=args.run_task)


def main():
    args = parse_args()
    setup(True)
    predict(args)


if __name__ == '__main__':
    main()
