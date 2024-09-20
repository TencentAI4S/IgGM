# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/5 17:08
import gzip
import os
from collections import OrderedDict
from functools import partial

from Bio import Seq
from Bio.SeqIO import FastaIO, SeqRecord


def parse_fasta(path, to_dict=False):
    """Parse the FASTA file.

    Args:
        path: path to the FASTA file (could be GZIP-compressed)

    Returns:
        prot_id: protein ID (as in the commentary line)
        aa_seq: amino-acid sequence
    """
    open_fn = partial(gzip.open, mode='rt') if path.endswith('.gz') else partial(open, mode='r')
    with open_fn(path) as f:
        fasta_string = f.read()

    return parse_fasta_string(fasta_string, to_dict=to_dict)


def parse_fasta_string(fasta_string, to_dict=False):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Args:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence ids
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    ids = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            seq_id, *description = line[1:].split(None, 1)  # Remove the '>' at the beginning.
            ids.append(seq_id)
            if len(description) > 0:
                descriptions.append(description)
            else:
                descriptions.append("")
            sequences.append('')
            continue
        elif line.startswith('#'):
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    assert len(sequences) == len(ids), f"unvalid fasta file"

    if to_dict:
        return OrderedDict(zip(ids, sequences))

    return sequences, ids, descriptions


def export_fasta(sequences, ids=None, output=None, descriptions=None):
    if len(sequences) == 0:
        return

    fh = None
    if output is not None:
        os.makedirs(os.path.dirname(os.path.realpath(output)), exist_ok=True)
        fh = open(output, "w")

    if ids is None:
        ids = [f"sequence{i}" for i in range(len(sequences))]

    if descriptions is None:
        descriptions = ["" for i in range(len(sequences))]

    fasta_string = []
    for seq, seq_id, desc in zip(sequences, ids, descriptions):
        fstring = FastaIO.as_fasta(SeqRecord(Seq.Seq(seq),
                                             id=seq_id,
                                             description=desc))
        if fh is not None:
            fh.write(fstring)
        else:
            fasta_string.append(fstring)

    if fh is not None:
        fh.close()
        return output
    else:
        return "".join(fasta_string)
