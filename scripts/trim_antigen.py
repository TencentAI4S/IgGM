#!/usr/bin/env python
"""
Standalone Antigen Trimming Tool.

Trims the specified antigen chain to epitope residues +/- neighborhood window.
All other chains (e.g., antibody H/L) are preserved unchanged.
Outputs both trimmed PDB and FASTA with renumbered antigen residues.

IMPORTANT: Epitope indices should be the same PDB residue numbers output by --cal_epitope.

Usage:
    python trim_antigen.py --pdb complex.pdb --fasta complex.fasta --output trimmed.pdb --antigen-chain A --epitope 198 199 200 --neighborhood 5
"""

import os
import sys
import argparse

# Add repo root to path to import IgGM
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Standard amino acid 3->1 letter mapping
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}

def parse_pdb_by_chain(pdb_file):
    """
    Parse a PDB file and extract residues organized by chain.
    Returns dict: {chain_id: [{'resnum': int, 'resname': str, 'atoms': [lines]}]}
    """
    chains = {}
    current_res = None
    current_chain = None
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                chain = line[21]
                res_num = int(line[22:26].strip())
                res_name = line[17:20].strip()
                
                if chain not in chains:
                    chains[chain] = []
                
                if current_chain != chain or current_res is None or current_res['resnum'] != res_num:
                    # New residue or new chain
                    if current_res is not None and current_chain is not None:
                        chains[current_chain].append(current_res)
                    current_res = {
                        'resnum': res_num,
                        'resname': res_name,
                        'atoms': [line]
                    }
                    current_chain = chain
                else:
                    current_res['atoms'].append(line)
    
    # Don't forget the last residue
    if current_res is not None and current_chain is not None:
        chains[current_chain].append(current_res)
    
    return chains

import math

def get_residue_ca(atom_lines):
    """Extract CA coordinate from atom lines. Returns (x, y, z) or None."""
    for line in atom_lines:
        if line[12:16].strip() == 'CA':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            return (x, y, z)
    return None

def trim_residues(residues, epitope_pdb_nums, neighborhood_size=5, keep_radius=0.0):
    """
    Trim residues to keep only epitope + neighborhood, PLUS residues within radius of CoM.
    Returns list of kept residues and the new renumbered epitope indices.
    """
    # Build mapping from resnum to sequential index
    all_resnums = [r['resnum'] for r in residues]
    resnum_to_idx = {num: idx for idx, num in enumerate(all_resnums)}
    
    # 1. Identify Seed Residues (Epitope + Neighborhood)
    seed_indices = set()
    for epi_num in epitope_pdb_nums:
        if epi_num in resnum_to_idx:
            idx = resnum_to_idx[epi_num]
            # Keep residues within neighborhood
            for i in range(max(0, idx - neighborhood_size), min(len(residues), idx + neighborhood_size + 1)):
                seed_indices.add(i)
    
    final_indices = seed_indices.copy()
    
    # 2. Radius Expansion (if enabled)
    if keep_radius > 0 and seed_indices:
        # Calculate CoM of seed residues
        seed_coords = []
        for idx in seed_indices:
            coord = get_residue_ca(residues[idx]['atoms'])
            if coord:
                seed_coords.append(coord)
        
        if seed_coords:
            n_seeds = len(seed_coords)
            com_x = sum(c[0] for c in seed_coords) / n_seeds
            com_y = sum(c[1] for c in seed_coords) / n_seeds
            com_z = sum(c[2] for c in seed_coords) / n_seeds
            
            print(f"  Center of Mass of seed residues: ({com_x:.1f}, {com_y:.1f}, {com_z:.1f})")
            print(f"  Expanding selection with radius {keep_radius}A...")
            
            # Check all residues
            sq_radius = keep_radius ** 2
            added_count = 0
            for i, res in enumerate(residues):
                if i in final_indices:
                    continue
                
                coord = get_residue_ca(res['atoms'])
                if coord:
                    dist_sq = (coord[0]-com_x)**2 + (coord[1]-com_y)**2 + (coord[2]-com_z)**2
                    if dist_sq <= sq_radius:
                        final_indices.add(i)
                        added_count += 1
            print(f"    Added {added_count} residues from radius check.")
    
    # Get residues to keep (in order)
    kept_residues = [residues[i] for i in sorted(final_indices)]
    
    # Calculate new epitope indices (1-based)
    kept_resnums = {r['resnum'] for r in kept_residues}
    new_epitope = []
    for new_idx, res in enumerate(kept_residues):
        if res['resnum'] in epitope_pdb_nums:
            new_epitope.append(new_idx + 1)
    
    return kept_residues, new_epitope

def write_pdb(chains_data, output_file, chain_order=None):
    """
    Write chains to PDB file.
    chains_data: dict {chain_id: [residues]}
    chain_order: optional list specifying order of chains
    """
    if chain_order is None:
        chain_order = sorted(chains_data.keys())
    
    with open(output_file, 'w') as f:
        atom_num = 1
        for chain_id in chain_order:
            if chain_id not in chains_data:
                continue
            residues = chains_data[chain_id]
            for new_resnum, res in enumerate(residues, start=1):
                for atom_line in res['atoms']:
                    # Renumber atoms and residues
                    new_line = (
                        atom_line[:6] +
                        f'{atom_num:5d}' +
                        atom_line[11:21] +
                        chain_id +
                        f'{new_resnum:4d}' +
                        atom_line[26:]
                    )
                    f.write(new_line)
                    atom_num += 1
            f.write('TER\n')
        f.write('END\n')

def write_fasta(chains_data, output_file, chain_order=None):
    """Write all chains to FASTA file."""
    if chain_order is None:
        chain_order = sorted(chains_data.keys())
    
    with open(output_file, 'w') as f:
        for chain_id in chain_order:
            if chain_id not in chains_data:
                continue
            residues = chains_data[chain_id]
            seq = ''.join(AA_3TO1.get(r['resname'], 'X') for r in residues)
            f.write(f'>{chain_id}\n{seq}\n')

def main():
    parser = argparse.ArgumentParser(description="IgGM Antigen Trimmer - trims antigen while preserving other chains")
    parser.add_argument('--pdb', required=True, help="Input complex PDB file")
    parser.add_argument('--fasta', required=True, help="Input FASTA file (for reference)")
    parser.add_argument('--output', required=True, help="Output trimmed complex PDB")
    parser.add_argument('--antigen-chain', type=str, default="A", help="Antigen Chain ID to trim (default A)")
    parser.add_argument('--neighborhood', type=int, default=5, help="Residues to keep on each side (default 5)")
    parser.add_argument('--keep-radius', type=float, default=25.0, help="Keep residues within this radius (A) of selection CoM (default 25.0)")
    parser.add_argument('--epitope', nargs='+', type=int, required=True, help="Epitope PDB residue numbers from --cal_epitope")
    
    args = parser.parse_args()
    
    # Derive output FASTA path
    output_fasta = args.output.replace('.pdb', '.fasta')
    
    # 1. Parse all chains from PDB
    print(f"Parsing PDB: {args.pdb}")
    chains = parse_pdb_by_chain(args.pdb)
    print(f"Found chains: {list(chains.keys())}")
    
    for chain_id, residues in chains.items():
        print(f"  Chain {chain_id}: {len(residues)} residues (PDB range: {residues[0]['resnum']} to {residues[-1]['resnum']})")
    
    # 2. Validate antigen chain exists
    antigen_chain = args.antigen_chain
    if antigen_chain not in chains:
        print(f"ERROR: Antigen chain '{antigen_chain}' not found in PDB!")
        print(f"Available chains: {list(chains.keys())}")
        sys.exit(1)
    
    antigen_residues = chains[antigen_chain]
    
    # 3. Validate epitope residues
    epitope_set = set(args.epitope)
    resnum_set = {r['resnum'] for r in antigen_residues}
    valid_epitope = [e for e in args.epitope if e in resnum_set]
    missing = [e for e in args.epitope if e not in resnum_set]
    
    if missing:
        print(f"WARNING: These epitope residues not found in chain {antigen_chain}: {missing}")
    
    if not valid_epitope:
        print("ERROR: No valid epitope residues found in antigen chain!")
        sys.exit(1)
    
    print(f"\nTrimming antigen chain {antigen_chain}:")
    print(f"  Epitope residues: {' '.join(map(str, valid_epitope))}")
    print(f"  Neighborhood: ±{args.neighborhood} residues")
    
    # 4. Trim antigen
    trimmed_antigen, new_epitope = trim_residues(antigen_residues, valid_epitope, args.neighborhood, args.keep_radius)
    print(f"  Trimmed: {len(antigen_residues)} → {len(trimmed_antigen)} residues")
    
    # 5. Build output chains (preserve all, replace antigen with trimmed version)
    output_chains = {}
    for chain_id, residues in chains.items():
        if chain_id == antigen_chain:
            output_chains[chain_id] = trimmed_antigen
        else:
            output_chains[chain_id] = residues
    
    # Determine chain order (put antigen last, as is convention)
    other_chains = [c for c in chains.keys() if c != antigen_chain]
    chain_order = sorted(other_chains) + [antigen_chain]
    
    # 6. Save PDB (with all chains)
    print(f"\nSaving trimmed PDB to {args.output}...")
    write_pdb(output_chains, args.output, chain_order)
    
    # 7. Save FASTA (with all chains)
    print(f"Saving FASTA to {output_fasta}...")
    write_fasta(output_chains, output_fasta, chain_order)
    
    # 8. Save antigen-only version (_noAb.pdb)
    output_noab_pdb = args.output.replace('.pdb', '_noAb.pdb')
    antigen_only = {antigen_chain: trimmed_antigen}
    
    print(f"Saving antigen-only PDB to {output_noab_pdb}...")
    write_pdb(antigen_only, output_noab_pdb, [antigen_chain])
    
    # Summary
    print(f"\nOutput summary:")
    for chain_id in chain_order:
        residues = output_chains[chain_id]
        seq = ''.join(AA_3TO1.get(r['resname'], 'X') for r in residues)
        status = "(TRIMMED)" if chain_id == antigen_chain else "(unchanged)"
        print(f"  Chain {chain_id}: {len(residues)} aa {status}")
    
    print(f"\nNew epitope indices for design: {' '.join(map(str, new_epitope))}")

if __name__ == "__main__":
    main()
