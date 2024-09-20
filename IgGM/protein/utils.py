# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import random

import torch

from .prot_constants import RESD_NAMES_1C, N_ANGLS_PER_RESD
from ..utils.diff_util import cord2fram_batch, rota2quat


def get_mlm_masks(aa_seqs, mask_prob=0.15, mask_vecs=None):
    """Get random masks for masked language modeling.

    Notes:
        For identical amino-acid sequences, their random masks will be the same, thus ensuring no
        label leakage for masked language modeling.

    Args:
        aa_seqs: list of amino-acid sequences, each of length L_i
        mask_prob: (optional) how likely one token is masked out
        mask_vecs: (optional) list of masking-allowed-or-not indicators, each of length L_i

    Returns:
        mask_vec: random masks of size L (L = \sum_i L_i)
    """
    # generate random masks for each unique amino-acid sequence
    mask_vec_dict = {}
    for seq_idx, aa_seq in enumerate(aa_seqs):
        if aa_seq in mask_vec_dict:  # do not re-generate random masks for the same sequence
            continue

        if mask_vecs is None:
            idxs_resd_cand = list(range(len(aa_seq)))  # all the residues are candidates
        else:
            idxs_resd_cand = torch.nonzero(mask_vecs[seq_idx])[:, 0].tolist()

        n_resds_cand = len(idxs_resd_cand)  # number of candidate residues
        n_resds_mask = int(n_resds_cand * mask_prob + 0.5)  # number of masked residues
        idxs_resd = random.sample(idxs_resd_cand, n_resds_mask)  # indices of masked residues
        mask_vec = torch.zeros(len(aa_seq), dtype=torch.int8)
        mask_vec[idxs_resd] = 1  # 1: masked-out token
        mask_vec_dict[aa_seq] = mask_vec

    # concatenate random masks into one
    mask_vec = torch.cat([mask_vec_dict[x] for x in aa_seqs], dim=0)

    return mask_vec


def apply_mlm_masks(tokn_mat_orig, mask_mat, alphabet):
    """Apply random masks for masked language modeling.

    Notes:
        For each token to be masked, it has an 80% probability of being replaced with a mask token,
        a 10% probability of being replaced with a random token (20 standard AA types), and an 10%
        probability of being replaced with an unmasked token.

    Args:
        tokn_mat_orig: original token indices of size N x L
        mask_mat: random masks of size N x L
        alphabet: alphabet used for tokenization

    Returns:
        tokn_mat_pert: perturbed token indices of size N x L
    """
    dtype = tokn_mat_orig.dtype
    device = tokn_mat_orig.device
    batch_size, seq_len = tokn_mat_orig.shape

    # generate perturbed token indices
    toks = random.choices(RESD_NAMES_1C, k=(batch_size * seq_len))
    prob_mat = torch.rand((batch_size, seq_len), dtype=torch.float32, device=device)
    mask_mat_pri = (mask_mat * torch.lt(prob_mat, 0.8)).to(torch.bool)
    mask_mat_sec = (mask_mat * torch.gt(prob_mat, 0.9)).to(torch.bool)
    tokn_mat_pri = alphabet.mask_idx * torch.ones_like(tokn_mat_orig)
    tokn_mat_sec = torch.tensor(
        [alphabet.get_idx(x) for x in toks], dtype=dtype, device=device).view(batch_size, seq_len)
    tokn_mat_pert = torch.where(
        mask_mat_pri, tokn_mat_pri, torch.where(mask_mat_sec, tokn_mat_sec, tokn_mat_orig))

    return tokn_mat_pert




def init_qta_params(n_smpls, n_resds, mode='black-hole', cord_tns=None, cmsk_tns=None):
    """Initialize quaternion-translation-angle (QTA) parameters.

    Args:
    * n_smpls: number of samples (denoted as N)
    * n_resds: number of residues (denoted as L)
    * mode: initialization mode (choices: 'black-hole' / 'random' / '3d-cord')
    * cord_tns: (optional) per-atom 3D coordinates of size N x L x 3 x 3 (N - CA - C)
    * cmsk_tns: (optional) per-atom 3D coordinates' validness masks of size N x L x 3 (N - CA - C)

    Returns:
    * quat_tns: quaternion vectors of size N x L x 4
    * trsl_tns: translation vectors of size N x L x 3
    * angl_tns: torsion angle matrices of size N x L x K x 2 (K = 7)

    Notes:
    * Currently, we only support 3D coordinates in the 'n3' format.
    """
    if mode == 'black-hole':
        quat_tns = torch.cat([
            torch.ones((n_smpls, n_resds, 1)),  # qr
            torch.zeros((n_smpls, n_resds, 3)),  # qx / qy / qz
        ], dim=2)
        trsl_tns = torch.zeros((n_smpls, n_resds, 3))
        angl_tns = torch.cat([
            torch.ones((n_smpls, n_resds, N_ANGLS_PER_RESD, 1)),  # cosine
            torch.zeros((n_smpls, n_resds, N_ANGLS_PER_RESD, 1)),  # sine
        ], dim=3)  # cos(x) = 1 & sin(x) = 0 => zero-initialization
    elif mode == '3d-cord':
        # validate 3D coordinates & validness masks
        assert list(cord_tns.shape) == [n_smpls, n_resds, 3, 3], f'unexpected shape in <cord_tns>: {cord_tns.shape}'
        assert list(cmsk_tns.shape) == [n_smpls, n_resds, 3], f'unexpected shape in <cmsk_tns>: {cmsk_tns.shape}'
        n_frams = n_smpls * n_resds

        # build per-residue backbone frames
        rota_tns, trsl_mat = cord2fram_batch(cord_tns.view(n_frams, 3, 3))
        quat_tns = rota2quat(rota_tns, quat_type='full').view(n_smpls, n_resds, 4)
        trsl_tns = trsl_mat.view(n_smpls, n_resds, 3)
        angl_tns = torch.cat([
            torch.ones((n_smpls, n_resds, N_ANGLS_PER_RESD, 1)),  # cosine
            torch.zeros((n_smpls, n_resds, N_ANGLS_PER_RESD, 1)),  # sine
        ], dim=3)  # cos(x) = 1 & sin(x) = 0 => zero-initialization

        # fall back to black-hole initialization for residues w/ missing atoms
        rmsk_tns = torch.all(cmsk_tns, dim=2, keepdim=True)  # N x L x 1
        quat_tns_bh, trsl_tns_bh, _ = init_qta_params(n_smpls, n_resds, mode='black-hole')
        quat_tns = torch.where(rmsk_tns, quat_tns, quat_tns_bh.to(quat_tns))
        trsl_tns = torch.where(rmsk_tns, trsl_tns, trsl_tns_bh.to(trsl_tns))
    elif mode == 'random':
        quat_tns = torch.randn((n_smpls, n_resds, 4))
        # quat_tns *= torch.sign(quat_tns[:, :, :1])  # qr: non-negative
        quat_tns[quat_tns[:, :, 0] < 0.0] *= -1  # qr: non-negative
        quat_tns /= torch.norm(quat_tns, dim=2, keepdim=True)  # unit L2-norm
        trsl_tns = torch.randn((n_smpls, n_resds, 3))
        angl_tns = torch.randn((n_smpls, n_resds, N_ANGLS_PER_RESD, 2))
        angl_tns /= torch.norm(angl_tns, dim=3, keepdim=True)  # unit L2-norm
    else:
        raise ValueError(f'unrecognized initialization mode for QTA parameters: {mode}')

    return quat_tns, trsl_tns, angl_tns

