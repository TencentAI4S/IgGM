"""Utility functions."""
import torch

from IgGM.protein import ProtStruct
from IgGM.utils import cdist


def gen_contact(chn_data_pri, chn_data_sec, inter_chain_contact, dist_thres=10):
    """Generate contact feature with two predicted coordinate and inter-chain contact feature."""

    cord_mat_pri = ProtStruct.get_atoms(chn_data_pri['seq'], chn_data_pri['cord'], ['CA'])
    cmsk_vec_pri = ProtStruct.get_cmsk_vld(chn_data_pri['seq'], cord_mat_pri.device)
    cmsk_vec_pri = ProtStruct.get_atoms(chn_data_pri['seq'], cmsk_vec_pri, ['CA'])

    cord_mat_sec = ProtStruct.get_atoms(chn_data_sec['seq'], chn_data_sec['cord'], ['CA'])
    cmsk_vec_sec = ProtStruct.get_cmsk_vld(chn_data_sec['seq'], cord_mat_sec.device)
    cmsk_vec_sec = ProtStruct.get_atoms(chn_data_pri['seq'], cmsk_vec_sec, ['CA'])

    len_pri = len(chn_data_pri['seq'])

    dist_mat = cdist(cord_mat_pri, cord_mat_pri)
    dist_max = torch.max(dist_mat)
    dist_mat += dist_max * (1 - torch.outer(cmsk_vec_pri, cmsk_vec_pri))
    contacts_pri = torch.lt(dist_mat, dist_thres)

    dist_mat = cdist(cord_mat_sec, cord_mat_sec)
    dist_max = torch.max(dist_mat)
    dist_mat += dist_max * (1 - torch.outer(cmsk_vec_sec, cmsk_vec_sec))
    contacts_sec = torch.lt(dist_mat, dist_thres)

    contact = inter_chain_contact.detach().clone()
    contact[0, :len_pri, :len_pri, 0] += contacts_pri
    contact[0, len_pri:, len_pri:, 0] += contacts_sec

    return contact


def contact_to_ppi(ligand_feat, receptor_feat, contact_data):
    """generate ppi data with contact data."""

    ligand_len = len(ligand_feat['seq'])
    inter_chain_contact = contact_data[0, :ligand_len, ligand_len:, 0]  # (ligand_len, receptor_len)
    ppi_pri = torch.where(torch.max(inter_chain_contact, dim=1)[0] == 1, 1, 0)
    ppi_sec = torch.where(torch.max(inter_chain_contact, dim=0)[0] == 1, 1, 0)
    ppi_data = torch.cat([ppi_pri, ppi_sec], dim=0).unsqueeze(0).unsqueeze(-1)

    return ppi_data


def ppi_to_contact(ligand_feat, receptor_feat, ppi_data):
    """generate pseudo contact using ppi data."""

    ligand_len = len(ligand_feat['seq'])
    receptor_len = len(receptor_feat['seq'])
    inter_chain_contact = torch.zeros([ligand_len + receptor_len, ligand_len + receptor_len, 1]).to(ppi_data.device)
    idxs_ppi = torch.where(ppi_data[0, :, 0] == 1)[0]
    inter_chain_contact[idxs_ppi, :, 0] = 1
    inter_chain_contact[:, idxs_ppi, 0] = 1
    inter_chain_contact = inter_chain_contact.unsqueeze(0)
    contact_data = gen_contact(ligand_feat, receptor_feat, inter_chain_contact)

    return contact_data
