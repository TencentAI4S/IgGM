# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import argparse
import os
import sys

import torch
import tqdm

from IgGM.protein import cal_ppi

sys.path.append('.')

from IgGM.deploy import AbDesigner
from IgGM.utils import setup
from IgGM.protein.parser import parse_fasta, PdbParser
from IgGM.model.pretrain import esm_ppi_650m_ab, antibody_design_trunk, IGSO3Buffer_trunk


def parse_args():
    parser = argparse.ArgumentParser(description='Antibody sequence and structure co-design w/ IgGM')
    parser.add_argument('--fasta', '-f', type=str, required=True, help='Directory path to input antibody FASTA files, X for design region')
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
        '--num_samples', '-ns',
        type=int,
        default=1,
        help='number of samples for each input',
    )
    args = parser.parse_args()

    return args


def predict(args):
    """Predict antibody & antigen sequence and structures w/ pre-trained IgGM-Ag models."""
    fasta_path = args.fasta
    pdb_path = args.antigen

    sequences, ids, _ = parse_fasta(fasta_path)
    assert len(sequences) in (1, 2, 3), f"must be 1, 2 or 3 chains in fasta file"
    chains = [{"sequence": seq, "id": seq_id} for seq, seq_id in zip(sequences, ids) if seq_id != "A"]
    _, basename = os.path.split(fasta_path)
    name = basename.split(".")[0]
    output = f"{args.output}/{name}.pdb"

    aa_seq, atom_cord, atom_cmsk, _, _ = PdbParser.load(pdb_path, chain_id="A")
    if args.epitope is None:
        try:
            epitope = cal_ppi(pdb_path, ids)
        except:
            epitope = args.epitope
    else:
        epitope = torch.zeros(len(aa_seq))
        for i in args.epitope:
            epitope[i] = 1
    chains.append({"sequence": aa_seq,
                    "cord": atom_cord,
                    "cmsk": atom_cmsk,
                    "epitope": epitope,
                    "id": "A"})


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
        design_path=antibody_design_trunk(),
        buffer_path=IGSO3Buffer_trunk(),
        config=args,
    )
    designer.to(device)

    chunk_size = args.chunk_size
    print(f"#inference samples: {len(batches)}")
    for task in tqdm.tqdm(batches):
        designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size)


def main():
    args = parse_args()
    setup(True)
    predict(args)


if __name__ == '__main__':
    main()
