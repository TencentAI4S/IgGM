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
        '--relax_open', '-r_open',
        action='store_true',
        help='relax structures after design using OpenMM',
    )
    # Restoration arguments
    parser.add_argument('--restore_merged', type=str, default=None, help='Path to merged antigen PDB (used for alignment)')
    parser.add_argument('--restore_unmerged', type=str, default=None, help='Path to original unmerged antigen PDB')
    parser.add_argument('--restore_IDs', type=str, default=None, help='Chain IDs in original PDB to restore (e.g., A_B_C)')
    parser.add_argument('--keep_trimmed', action='store_true', help='Keep trimmed sequences in restored output')
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
        designer.infer_pdb(task["chains"], filename=task["output"], chunk_size=chunk_size, relax=args.relax, relax_open=args.relax_open, temperature=temperature, task=args.run_task)


def main():
    args = parse_args()
    setup(True)
    # Logic change: If restoration is requested, we disable generation-time relaxation
    # and perform it AFTER restoration on the full complex.
    do_restore = True if (args.restore_merged and args.restore_unmerged and args.restore_IDs) else False
    
    saved_relax = args.relax
    saved_relax_open = args.relax_open
    
    if do_restore:
        print("[Design] Restoration requested: Deferring relaxation until after restoration.")
        args.relax = False
        args.relax_open = False

    predict(args)


    
    # Post-inference restoration
    if args.restore_merged and args.restore_unmerged and args.restore_IDs:
        try:
            print("[Restore] Starting antigen restoration...")
            import subprocess
            import os
            import sys
            
            # Identify output files from the previous step
            output_dir = os.path.dirname(args.output) if args.output.endswith('.pdb') else args.output
            if not output_dir: output_dir = '.'
            
            # Derive base name from input FASTA
            input_base = os.path.basename(args.fasta).replace('.fasta', '')
            base_name = input_base
            
            for i in range(args.num_samples):
                # Standard naming convention: {output_dir}/{input_base}_{i}.pdb
                designed_pdb = f"{output_dir}/{base_name}_{i}.pdb"
                
                if not os.path.exists(designed_pdb):
                     # Fallback check
                     designed_pdb = f"{args.output}_{i}.pdb"
                
                if os.path.exists(designed_pdb):
                    restored_pdb = designed_pdb.replace('.pdb', '_restored.pdb')
                    
                    cmd = [
                        sys.executable,
                        f"{os.path.dirname(os.path.abspath(__file__))}/scripts/restore_antigen.py",
                        "--designed-pdb", designed_pdb,
                        "--merged-pdb", args.restore_merged,
                        "--original-pdb", args.restore_unmerged,
                        "--restore-ids", args.restore_IDs,
                        "--output", restored_pdb
                    ]
                    
                    print(f"[Restore] Running: {' '.join(cmd)}")
                    subprocess.check_call(cmd)
                    
                    # Perform Deferred Relaxation on Restricted PDB
                    if do_restore and (saved_relax or saved_relax_open):
                        print(f"[Restore] Relaxing restored structure: {restored_pdb}")
                        from IgGM.utils import OpenMM_relax, Rosetta_relax
                        
                        if saved_relax_open:
                            try:
                                print(f"[Restore] Running OpenMM Relax on {restored_pdb}...")
                                OpenMM_relax(restored_pdb)
                                
                                # OpenMM/PDBFixer often renames chains (A, B, C...). 
                                # We restore the original chain IDs (Antigen chains + H + L) based on the known input order.
                                # Assumes OpenMM preserves the order of chains.
                                
                                print(f"[Restore] Restoring chain IDs after OpenMM relaxation...")
                                restored_ids = args.restore_IDs.split('_') # e.g. ['A', 'B', 'C']
                                ab_ids = ['H', 'L'] # Standard IgGM output
                                expected_order = restored_ids + ab_ids
                                
                                from Bio.PDB import PDBParser, PDBIO
                                parser = PDBParser(QUIET=True)
                                st = parser.get_structure("relaxed", restored_pdb)
                                
                                model = st[0]
                                chains = list(model.get_chains())
                                
                                if len(chains) == len(expected_order):
                                    for i, chain in enumerate(chains):
                                        new_id = expected_order[i]
                                        print(f"      Renaming chain {chain.id} -> {new_id}")
                                        chain.id = new_id
                                        
                                    io = PDBIO()
                                    io.set_structure(st)
                                    io.save(restored_pdb)
                                    print(f"[Restore] Chain IDs restored in {restored_pdb}")
                                else:
                                    print(f"      [WARNING] Chain count mismatch (Found {len(chains)}, Expected {len(expected_order)}). Skipping rename.")
                                    
                            except Exception as e:
                                print(f"[Restore] OpenMM Relax/Rename failed: {e}")
                                
                        if saved_relax:
                            try:
                                print(f"[Restore] Running Rosetta Relax on {restored_pdb}...")
                                # Rosetta_relax usually overwrites or creates a numbered suffix? 
                                # IgGM.utils.Rosetta_relax implementation:
                                # def Rosetta_relax(pdb_file): ... tries to relax and overwrite/save.
                                Rosetta_relax(restored_pdb)
                            except Exception as e:
                                print(f"[Restore] Rosetta Relax failed: {e}")
                    
                    # Delete trimmed output if not keeping
                    if not args.keep_trimmed:
                        print(f"[Restore] Deleting trimmed output: {designed_pdb}")
                        os.remove(designed_pdb)
                        # Also remove fasta if exists
                        designed_fasta = designed_pdb.replace('.pdb', '.fasta')
                        if os.path.exists(designed_fasta):
                            os.remove(designed_fasta)
                            
        except Exception as e:
            print(f"[Restore] Error during restoration: {e}")

if __name__ == '__main__':
    main()
