# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch

from .se3 import quat2rot, rot2quat


def update_se3_trans(quat_tns_old, trsl_tns_old, quat_tns_upd, trsl_tns_upd):
    """Update SE(3) transformations.

    Args:
        quat_tns_old: old quaternion vectors of size N x L x 4
        trsl_tns_old: old translation vectors of size N x L x 3
        quat_tns_upd: update terms of quaternion vectors of size N x L x 4
        trsl_tns_upd: update terms of translation vectors of size N x L x 3

    Returns*
        quat_tns_new: new quaternion vectors of size N x L x 4
        trsl_tns_new: new translation vectors of size N x L x 3
    """

    # initialization
    n_smpls, n_resds, _ = quat_tns_old.shape

    # obtain previous rotation matrices and their update terms
    rota_tns_old = quat2rot(quat_tns_old[0]).unsqueeze(dim=0)  # N x L x 3 x 3
    rota_tns_upd = quat2rot(quat_tns_upd[0]).unsqueeze(dim=0)

    # obtain new rotation matrices
    rota_tns_new = torch.bmm(rota_tns_old[0], rota_tns_upd[0]).unsqueeze(dim=0)

    # obtain new quaternion & translation vectors
    quat_tns_new = rot2quat(rota_tns_new[0], quat_type='full').view(n_smpls, n_resds, -1)
    trsl_tns_new = trsl_tns_old + \
                   torch.bmm(rota_tns_old[0], trsl_tns_upd.view(-1, 3, 1)).view(n_smpls, n_resds, -1)

    return quat_tns_new, trsl_tns_new
