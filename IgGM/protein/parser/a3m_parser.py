# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/5 17:08
import string
from typing import Sequence, Tuple

from .fasta_parser import parse_fasta_string

DEL_LOWERCASE_TABLE = str.maketrans(dict.fromkeys(string.ascii_lowercase))

DeletionMatrix = Sequence[Sequence[int]]


def parse_a3m(a3m_string: str) -> Tuple[Sequence[str], DeletionMatrix]:
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
        a3m_string: The string contents of a a3m file. The first sequence in the
            file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
    """
    sequences, *_ = parse_fasta_string(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


def a3m_to_a2m(a3m_string):
    A2Ms = []
    for line in a3m_string.splitlines():
        if line[0] in ('>', '#'):
            continue
        line = line.rstrip()
        # lower?
        A2Ms.append(line.translate(DEL_LOWERCASE_TABLE))

    return '\n'.join(A2Ms)


def merge_a3ms(a3ms):
    merged_a3m = []
    seen_sequences = set()
    for a3m in a3ms:
        lines = a3m.splitlines()
        for i, line in enumerate(lines):
            if line.startswith('>'):
                seq_name, sequence = line.strip(), lines[i + 1].strip()
                if sequence in seen_sequences:
                    continue
                seen_sequences.add(sequence)
                merged_a3m.extend([seq_name, sequence])

    return '\n'.join(merged_a3m)
