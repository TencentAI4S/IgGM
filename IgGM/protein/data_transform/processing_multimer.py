# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/5 17:19
import torch

from IgGM.utils.tensor import cdist
from ..prot_struct import ProtStruct


def get_entity_ids(aa_seqs):
    """Get entity IDs.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
        enty_ids: entity IDs of size L (L = \sum_i L_i)
    """
    aa_seqs_uniq = sorted(list(set(aa_seqs)))
    seq_lens = torch.tensor([len(x) for x in aa_seqs], dtype=torch.int32)
    enty_ids = torch.tensor(
        [aa_seqs_uniq.index(x) for x in aa_seqs]).repeat_interleave(seq_lens) + 1

    return enty_ids


def get_asym_ids(aa_seqs):
    """Get asymmetric IDs.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
        asym_ids: asymmetric IDs of size L (L = \sum_i L_i)
    """

    seq_lens = torch.tensor([len(x) for x in aa_seqs], dtype=torch.int32)
    asym_ids = torch.arange(len(aa_seqs)).repeat_interleave(seq_lens) + 1

    return asym_ids


def calc_ppi_sites(prot_data, chn_ids, dist_thres=10):
    """Calculate PPI sites"""

    assert len(chn_ids) == 2

    chn_data_pri = prot_data[chn_ids[0]]
    chn_data_sec = prot_data[chn_ids[1]]
    cord_mat_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cord'], ['CA'])
    cmsk_vec_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cmsk'], ['CA'])
    cord_mat_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cord'], ['CA'])
    cmsk_vec_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cmsk'], ['CA'])
    dist_mat = cdist(cord_mat_pri, cord_mat_sec)
    dist_max = torch.max(dist_mat)
    dist_mat += dist_max * (1 - torch.outer(cmsk_vec_pri, cmsk_vec_sec))

    dist_vec_pri = torch.min(dist_mat, dim=1)[0]  # minimal distance to the secondary chain
    dist_vec_sec = torch.min(dist_mat, dim=0)[0]  # minimal distance to the primary chain
    ppi_data = {}
    ppi_data[chn_ids[0]] = torch.lt(dist_vec_pri, dist_thres).to(torch.int8)
    ppi_data[chn_ids[1]] = torch.lt(dist_vec_sec, dist_thres).to(torch.int8)

    return ppi_data
