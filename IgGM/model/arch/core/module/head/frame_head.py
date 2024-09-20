# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import torch
import torch.nn as nn

from IgGM.protein.prot_constants import RESD_NAMES_1C, N_ANGLS_PER_RESD
from IgGM.transform import update_se3_trans
from .angle_head import AngleHead


class FrameHead(nn.Module):
    """The local frame prediction network."""

    def __init__(self, c_s, c_h):
        super().__init__()
        self.c_s = c_s
        self.c_h = c_h
        self.quat_dim = 4
        self.trans_dim = 3
        self.net = nn.ModuleDict()
        self.net['linear-q'] = nn.Linear(self.c_s, self.quat_dim)
        self.net['linear-t'] = nn.Linear(self.c_s, self.trans_dim)

    def forward(self, s, quat_tns_old, trsl_tns_old):
        """
        Args:
            s: single features & positional encodings of size N x L x (D_s + D_e)
            quat_tns_old: old quaternion vectors of size N x L x 4
            trsl_tns_old: old translation vectors of size N x L x 3

        Returns:
            quat_tns_new: new quaternion vectors of size N x L x 4
            trsl_tns_new: new translation vectors of size N x L x 3
            quat_tns_upd: update signals of quaternion vectors of size N x L x 4
        """
        # update quaternion & translation vectors
        if s.dtype != self.net['linear-q'].weight.dtype:
            s = s.to(self.net['linear-q'].weight.dtype)
        if quat_tns_old.dtype != self.net['linear-q'].weight.dtype:
            quat_tns_old = quat_tns_old.to(self.net['linear-q'].weight.dtype)
        if trsl_tns_old.dtype != self.net['linear-t'].weight.dtype:
            trsl_tns_old = trsl_tns_old.to(self.net['linear-t'].weight.dtype)
        quat_tns_upd = self.net['linear-q'](s)
        trsl_tns_upd = self.net['linear-t'](s)
        quat_tns_new, trsl_tns_new = update_se3_trans(
            quat_tns_old, trsl_tns_old, quat_tns_upd, trsl_tns_upd)

        return quat_tns_new, trsl_tns_new, quat_tns_upd


class FrameAngleHead(nn.Module):  # pylint: disable=too-many-instance-attributes
    """The network for updating per-residue local frames & torsion angles."""

    def __init__(
            self,
            c_s=384,
            n_dims_encd=32,  # number of dimensiosn in positional encodings
            n_dims_hidd=128,  # number of dimensions in hidden features
            decouple_angle=True  # whether to predict torsion angles w/ AA-type dependent networks
    ):
        super().__init__()
        self.c_s = c_s
        self.n_dims_encd = n_dims_encd
        self.n_dims_hidd = n_dims_hidd
        self.decouple_angle = decouple_angle
        self.n_dims_sfcd = self.c_s + self.n_dims_encd
        self.n_dims_angl = 2  # cosine & sine values for each torsion angle
        self.fram_net = FrameHead(self.n_dims_sfcd, self.n_dims_hidd)
        if not self.decouple_angle:
            self.angl_net = AngleHead(self.n_dims_sfcd, self.n_dims_hidd)
        else:
            self.angl_net = nn.ModuleDict()
            for resd_name in RESD_NAMES_1C:
                self.angl_net[resd_name] = AngleHead(self.n_dims_sfcd, self.n_dims_hidd)

    def forward(self, aa_seq, sfea_tns, sfea_tns_init, encd_tns, quat_tns,
                trsl_tns):
        """
        Args:
            aa_seq: amino-acid sequence
            sfea_tns: single features of size N x L x D_s
            sfea_tns_init: initial single features of size N x L x D_s
            encd_tns: positional encodings of size N x L x D_e
            quat_tns: old quaternion vectors of size N x L x 4
            trsl_tns: old translation vectors of size N x L x 3

        Returns:
            quat_tns: new quaternion vectors of size N x L x 4
            trsl_tns: new translation vectors of size N x L x 3
            angl_tns: torsion angle matrices of size N x L x K x 2
            quat_tns_upd: update signal of quaternion vectors of size N x L x 4
        """
        n_smpls, n_resds, _ = sfea_tns.shape
        dtype, device = sfea_tns.dtype, sfea_tns.device

        # concatenate single features & positional encodings
        sfcd_tns = torch.cat([sfea_tns, encd_tns], dim=2)
        sfcd_tns_init = torch.cat([sfea_tns_init, encd_tns], dim=2)

        # update per-residue local frames
        quat_tns, trsl_tns, quat_tns_upd = self.fram_net(sfcd_tns, quat_tns, trsl_tns)

        # predict per-residue torsion angles
        if not self.decouple_angle:
            angl_tns = self.angl_net(sfcd_tns, sfcd_tns_init)
        else:
            angl_tns = torch.zeros(
                (n_smpls, n_resds, N_ANGLS_PER_RESD, self.n_dims_angl), dtype=dtype, device=device)
            for resd_name in RESD_NAMES_1C:
                idxs = [idx for idx, name in enumerate(aa_seq) if name == resd_name]
                if len(idxs) != 0:
                    angl_tns[:, idxs] = \
                        self.angl_net[resd_name](sfcd_tns[:, idxs], sfcd_tns_init[:, idxs])

        return quat_tns, trsl_tns, angl_tns, quat_tns_upd
