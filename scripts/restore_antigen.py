#!/usr/bin/env python
"""
Antigen Restoration Tool.

Aligns the designed antibody back to the full, original antigen structure by calculating
structural alignment between the trimmed design antigen and the original antigen.

Flow:
1. Align Designed_Antigen -> Merged_Antigen (T1)
2. Align Merged_Antigen -> Original_Antigen (T2)
3. Apply T = T2 * T1 to Antibody chains
4. Save Antibody + Original Antigen

Usage:
    python restore_antigen.py ...
"""

import os
import sys
import argparse
import numpy as np
from Bio import pairwise2
from Bio.PDB import PDBParser, Superimposer, PDBIO, Select

def get_chain_data(structure, chain_id):
    """Extract CA atoms and Sequence for a chain."""
    atoms = []
    seq = ""
    # Map 3-letter codes to 1-letter codes
    aa_map = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if 'CA' in residue:
                        atoms.append(residue['CA'])
                        seq += aa_map.get(residue.resname, 'X')
    return atoms, seq

def get_all_ca_atoms(structure, chain_ids=None):
    """Get all CA atoms for specified chains (or all if None)."""
    atoms = []
    for model in structure:
        for chain in model:
            if chain_ids is None or chain.id in chain_ids:
                for residue in chain:
                    if 'CA' in residue:
                        atoms.append(residue['CA'])
    return atoms

def align_subsequence(query_atoms, query_seq, target_atoms, target_seq):
    """
    Find best local alignment of query within target and return paired atoms.
    
    This handles potential gaps or discontinuous segments (e.g., radius-trimmed antigens).
    Uses Biopython's pairwise2 local alignment (match=+5, mismatch=-5, gap_open=-10, gap_extend=-0.5).
    """
    # Perform local alignment to identifying matching subsequences using a robust scoring scheme.
    
    alignments = pairwise2.align.globalms(target_seq, query_seq, 5, -5, -10, -0.5)
    best = alignments[0]
    aligned_target, aligned_query, _, _, _ = best
    
    # Map atoms
    # Iterate through alignment.
    # If both align (not '-'), map target_atom[i] <-> query_atom[j]
    
    target_indices = []
    query_indices = []
    
    t_idx = 0
    q_idx = 0
    
    valid_pairs_target = []
    valid_pairs_query = []
    
    for t_char, q_char in zip(aligned_target, aligned_query):
        if t_char != '-':
            curr_t_atom = target_atoms[t_idx] if t_idx < len(target_atoms) else None
            t_idx += 1
        else:
            curr_t_atom = None
            
        if q_char != '-':
            curr_q_atom = query_atoms[q_idx] if q_idx < len(query_atoms) else None
            q_idx += 1
        else:
            curr_q_atom = None
            
        if t_char != '-' and q_char != '-' and t_char == q_char:
            valid_pairs_target.append(curr_t_atom)
            valid_pairs_query.append(curr_q_atom)
            
    return valid_pairs_query, valid_pairs_target # Matching pair lists

def main():
    parser = argparse.ArgumentParser(description="IgGM Antigen Restoration Post-Inference")
    parser.add_argument('--designed-pdb', required=True, help="Input designed PDB (trimmed Ag + Ab)")
    parser.add_argument('--merged-pdb', required=True, help="Merged PDB used for design")
    parser.add_argument('--original-pdb', required=True, help="Original full PDB")
    parser.add_argument('--output', required=True, help="Output restored PDB")
    parser.add_argument('--antigen-chain', default="A", help="Chain ID of antigen in design/merged")
    parser.add_argument('--restore-ids', required=True, help="Chain IDs in original PDB to preserve")
    
    args = parser.parse_args()
    
    pdb_parser = PDBParser(QUIET=True)
    
    print(f"Loading structures...")
    design_st = pdb_parser.get_structure('design', args.designed_pdb)
    merged_st = pdb_parser.get_structure('merged', args.merged_pdb)
    original_st = pdb_parser.get_structure('original', args.original_pdb)
    
    # === Step 1: Align Designed -> Merged ===
    print("\n--- Alignment 1: Designed -> Merged ---")
    
    # Get atoms/seq for trimmed antigen in Design
    d_atoms, d_seq = get_chain_data(design_st, args.antigen_chain)
    print(f"Designed Antigen (Chain {args.antigen_chain}): {len(d_atoms)} atoms, {len(d_seq)} residues")
    
    # Get atoms/seq for full antigen in Merged structure.
    # The merged antigen is typically assigned chain 'A' during the preprocessing step.
    m_atoms, m_seq = get_chain_data(merged_st, args.antigen_chain)
    print(f"Merged Antigen (Chain {args.antigen_chain}): {len(m_atoms)} atoms, {len(m_seq)} residues")
    
    # Find alignment
    # If d_atoms is subset of m_atoms
    pairs_d, pairs_m = align_subsequence(d_atoms, d_seq, m_atoms, m_seq)
    print(f"Matched {len(pairs_d)} residues between Designed and Merged")
    
    if len(pairs_d) < 3:
        print("Error: Insufficient matching residues for alignment (<3)")
        sys.exit(1)
        
    # Calculate T1 (Designed -> Merged)
    sup1 = Superimposer()
    sup1.set_atoms(pairs_m, pairs_d) # Fixed, Moving
    # Apply transform to Design structure (moves Ab to Merged frame)
    print(f"Applying T1 (RMSD: {sup1.rms:.3f})...")
    sup1.apply(design_st.get_atoms())
    
    
    # === Step 2: Align Merged -> Original ===
    print("\n--- Alignment 2: Merged -> Original ---")
    
    # Map residues from the Merged antigen (referenced as Chain A) to the Original multi-chain structure.
    # Since the Merged sequence is a concatenation of the Original sequences, we align them to establish the coordinate transform.
    
    orig_chain_ids = args.restore_ids.split('_')
    print(f"Original chains: {orig_chain_ids}")
    
    o_atoms = get_all_ca_atoms(original_st, orig_chain_ids)
    # Get sequence for all original CA atoms (might be discontinuous across chains, flattened)
    o_seq = ""
    aa_map = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    for atom in o_atoms:
        o_seq += aa_map.get(atom.get_parent().resname, 'X')
        
    print(f"Original Antigen (Chains {args.restore_ids}): {len(o_atoms)} atoms")
    
    # Align Merged (Moving) -> Original (Fixed) to calculate Transformation T2.
    # This transforms coordinates from the Merged frame to the Original PDB frame.
    
    pairs_m2, pairs_o = align_subsequence(m_atoms, m_seq, o_atoms, o_seq)
    print(f"Matched {len(pairs_m2)} residues between Merged and Original")
    
    if len(pairs_m2) < 3:
        print("Error: Insufficient match for Merged->Original alignment")
        # Proceed with identity? Warn user.
        pass
    else:
        sup2 = Superimposer()
        sup2.set_atoms(pairs_o, pairs_m2) # Fixed, Moving
        print(f"Applying T2 (RMSD: {sup2.rms:.3f})...")
        # Apply T2 to Design structure (which is already in Merged frame)
        sup2.apply(design_st.get_atoms())
        
        
    # === Step 3: Assembly ===
    print("\n--- Assembly ---")
    # Identify Antibody chains from Design (those != antigen_chain)
    ab_chains = [c for c in design_st[0] if c.id != args.antigen_chain]
    print(f"Antibody chains to save: {[c.id for c in ab_chains]}")
    
    # Identify Original chains
    orig_chains = [c for c in original_st[0] if c.id in orig_chain_ids]
    
    # Build output
    io = PDBIO()
    
    class OutputSelect(Select):
        def accept_chain(self, chain):
            return True
            
    # Construct a new Structure object for the final output.
    # This approach avoids issues with modifying existing Structure/Model parent pointers in Biopython.
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    
    out_st = Structure('output')
    out_model = Model(0)
    out_st.add(out_model)
    
    # Add Original Chains (A, B, C...)
    # They are already in correct frame (Fixed)
    for c in orig_chains:
        out_model.add(c.copy())
        
    # Add Antibody Chains (H, L...)
    # They have been transformed (T2 * T1)
    for c in ab_chains:
        out_model.add(c.copy())
        
    print(f"Saving to {args.output}...")
    io.set_structure(out_st)
    io.save(args.output)
    
    # FASTA
    output_fasta = args.output.replace('.pdb', '.fasta')
    with open(output_fasta, 'w') as f:
        for chain in out_model:
            seq = ""
            for res in chain:
                if 'CA' in res:
                    seq += aa_map.get(res.resname, 'X')
            f.write(f">{chain.id}\n{seq}\n")
    print("Done.")

if __name__ == "__main__":
    main()
