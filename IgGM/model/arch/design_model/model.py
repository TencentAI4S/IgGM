# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging

import torch
from torch import nn

from IgGM.model.layer import PPIEmbedding, ContactEmebedding, ChainRelativePositionEmbedding
from IgGM.model.layer.embedding import SinusoidalPositionEmbedding, RelativePositionEmbedding, StructEncoder, \
    RcEmbedNet
from IgGM.model.module.evoformer import EvoformerStackSS
from IgGM.protein.prot_constants import RESD_NAMES_1C
from ..base_model import BaseModel
from ..core.module import PairPredictor, StructureModule
from ...build import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DesignModel(BaseModel):
    """The antibody design model
    """

    def __init__(
            self,
            n_dims_sfea_init=-1,  # number of dimensions in initial single features (D_si)
            n_dims_pfea_init=-1,  # number of dimensions in initial pair features (D_pi)
            n_steps=200,  # number of time steps in the diffusion process
            n_dims_sfea=192,  # number of dimensions in single features (D_s)
            n_dims_pfea=128,  # number of dimensions in pair features (D_p)
            n_dims_penc=64,  # number of dimensions in positional encodings
            n_lyrs_2d=16,  # number of <EvoformerBlockSS> layers
            n_lyrs_3d=8,  # number of <AF2SMod> layers
            pred_oxyg=True,  # whether to predict backbone oxygen atoms' 3D coordinates
            use_icf=True,  # whether to use inter-chain interface feature as extra input
            feat_type='mix-4',  # inter-chain interface feature

    ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_sfea_init = n_dims_sfea_init
        self.n_dims_pfea_init = n_dims_pfea_init
        self.n_steps = n_steps
        self.n_dims_sfea = n_dims_sfea
        self.n_dims_pfea = n_dims_pfea
        self.n_dims_penc = n_dims_penc
        self.n_lyrs_2d = n_lyrs_2d
        self.n_lyrs_3d = n_lyrs_3d
        self.pred_oxyg = pred_oxyg


        # additional configurations
        self.n_bins = 18
        self.dist_min = 3.375
        self.dist_max = 21.375
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_bins

        # build various positional encoders
        self.ts_encoder = SinusoidalPositionEmbedding(self.n_dims_penc, max_len=n_steps)
        self.sq_encoder = SinusoidalPositionEmbedding(self.n_dims_penc, max_len=1024)  # max. 1024 residues
        self.rp_encoder = RelativePositionEmbedding(self.n_dims_pfea)
        self.st_encoder = StructEncoder(self.n_dims_pfea)

        # chain relative positional encoder
        self.crpe_encoder = ChainRelativePositionEmbedding(self.n_dims_pfea)

        self.use_icf = use_icf
        self.feat_type = feat_type
        if self.use_icf:
            self.icf_embed_sfea = PPIEmbedding(self.n_dims_sfea)
            # if self.feat_type == 'mix-3' or self.feat_type == 'mix-4':
            self.icf_embed_pfea = ContactEmebedding(self.n_dims_pfea)
        # build the model
        self.net = self.__build_model()

    @classmethod
    def restore(cls, path, config=None):
        """Restore a pre-trained model."""
        state = torch.load(path, map_location='cpu')
        logging.info(config)
        model = cls(n_dims_sfea_init=config.c_s, n_dims_pfea_init=config.c_p)
        model.load_state_dict(state, strict=False)
        logging.info('restore the pre-trained IgGM-Ag model %s', path)
        return model

    """The protein sequence & structure denoiser."""

    @property
    def n_params(self):
        """Get the number of model parameters."""

        return sum(x.numel() for x in self.parameters())

    @classmethod
    def featurize(cls, plm_featurizer, prot_data):
        """Build input features for protein data.

        Args:
        * plm_featurizer: PLM featurizer
        * prot_data: dict of protein data (as returned by <ProtDiffuser>)

        Returns:
        * inputs: dict of input tensors
        """
        asym_id = prot_data['asym-id'][0].view(-1)
        aa_len = len(asym_id)
        seqs = prot_data['seq-p']

        if asym_id.max() == 1:
            h_seqs = [''.join(seq[i] for i in range(aa_len) if asym_id[i] == 1) for seq in seqs]
            a_seqs = [''.join(seq[i] for i in range(aa_len) if asym_id[i] == 0) for seq in seqs]
            plm_outs = []
            for h_seq, a_seq in zip(h_seqs, a_seqs):
                aa_seq_ab = [h_seq]
                aa_seq_ag = [a_seq]
                with torch.no_grad():
                    plm_out_ab = plm_featurizer(aa_seq_ab)
                    plm_out_ag = plm_featurizer(aa_seq_ag)
                plm_outs.append((plm_out_ab, plm_out_ag))
            ab_sfea_mat_list = [x[0]['sfea'] for x in plm_outs]
            ab_pfea_tns_list = [x[0]['pfea'] for x in plm_outs]
            ag_sfea_mat_list = [x[1]['sfea'] for x in plm_outs]
            ag_pfea_tns_list = [x[1]['pfea'] for x in plm_outs]
            ab_sfea_mat, ab_pfea_tns = torch.cat(ab_sfea_mat_list, dim=0), torch.cat(ab_pfea_tns_list, dim=0)
            ag_sfea_mat, ag_pfea_tns = torch.cat(ag_sfea_mat_list, dim=0), torch.cat(ag_pfea_tns_list, dim=0)
            sfea_mat = torch.cat([ab_sfea_mat, ag_sfea_mat], dim=1)  # N x L x D_s
            pfea_tns = torch.zeros(ag_pfea_tns.shape[0], aa_len, aa_len, ag_pfea_tns.shape[-1]).type_as(ag_pfea_tns)
            pfea_tns[:, :ab_pfea_tns.shape[1], :ab_pfea_tns.shape[1], :] = ab_pfea_tns
            pfea_tns[:, -ag_pfea_tns.shape[1]:, -ag_pfea_tns.shape[1]:, :] = ag_pfea_tns
            del ab_sfea_mat_list, ab_pfea_tns_list, ag_sfea_mat_list, ag_pfea_tns_list
            inputs = {
                **prot_data,
                'sfea-i': sfea_mat,  # N x L x D_s
                'pfea-i': pfea_tns,  # N x L x L x D_p
            }
        elif asym_id.max() == 2:
            h_seqs = [''.join(seq[i] for i in range(aa_len) if asym_id[i] == 2) for seq in seqs]
            l_seqs = [''.join(seq[i] for i in range(aa_len) if asym_id[i] == 1) for seq in seqs]
            a_seqs = [''.join(seq[i] for i in range(aa_len) if asym_id[i] == 0) for seq in seqs]
            plm_outs = []
            for h_seq, l_seq, a_seq in zip(h_seqs, l_seqs, a_seqs):
                aa_seq_ab = [h_seq, l_seq]
                aa_seq_ag = [a_seq]
                with torch.no_grad():
                    plm_out_ab = plm_featurizer(aa_seq_ab)
                    plm_out_ag = plm_featurizer(aa_seq_ag)
                plm_outs.append((plm_out_ab, plm_out_ag))
            ab_sfea_mat_list = [x[0]['sfea'] for x in plm_outs]
            ab_pfea_tns_list = [x[0]['pfea'] for x in plm_outs]
            ag_sfea_mat_list = [x[1]['sfea'] for x in plm_outs]
            ag_pfea_tns_list = [x[1]['pfea'] for x in plm_outs]
            ab_sfea_mat, ab_pfea_tns = torch.cat(ab_sfea_mat_list, dim=0), torch.cat(ab_pfea_tns_list, dim=0)
            ag_sfea_mat, ag_pfea_tns = torch.cat(ag_sfea_mat_list, dim=0), torch.cat(ag_pfea_tns_list, dim=0)

            sfea_mat = torch.cat([ab_sfea_mat, ag_sfea_mat], dim=1)  # N x L x D_s
            pfea_tns = torch.zeros(ag_pfea_tns.shape[0], aa_len, aa_len, ag_pfea_tns.shape[-1]).type_as(ag_pfea_tns)
            pfea_tns[:, :ab_pfea_tns.shape[1], :ab_pfea_tns.shape[1], :] = ab_pfea_tns
            pfea_tns[:, -ag_pfea_tns.shape[1]:, -ag_pfea_tns.shape[1]:, :] = ag_pfea_tns
            del ab_sfea_mat_list, ab_pfea_tns_list, ag_sfea_mat_list, ag_pfea_tns_list
            inputs = {
                **prot_data,
                'sfea-i': sfea_mat,  # L x D_s
                'pfea-i': pfea_tns,  # L x L x D_p
            }
        else:
            raise ValueError('Invalid asym-id')
        torch.cuda.empty_cache()
        return inputs

    def forward(self, inputs, inputs_addi=None, chunk_size=None):
        """Perform the forward pass.

        Args:
        * inputs: dict of perturbed protein data ($x_{t}$)
        * inputs_addi: (optional) dict of additional protein data ($x_{t + 1}$ / $\hat{x}_{0}$)

        Returns:
        * outputs: dict of output tensors

        Notes:
        * The input dict of perturbed protein data should contain following entries:
          > step: list of time-step indices
          > pmsk: per-residue perturbed-or-not masks of size L
          > seq-p: list of perturbed amino-acid sequences, each of length L
          > cord-p: batched perturbed per-atom 3D coordinates of size N x L x M x 3
          > cmsk-p: batched perturbed per-atom 3D coordinates' validness masks of size N x L x M
          > sfea-i: initial single features of size N x L x D_si
          > pfea-i: initial pair features of size N x L x L x D_pi
        * Additional protein data can be either $x_{t + 1}$ (in this case, $x_{t}$ should be an
            interpolation between original protein data $x_{0}$ and $x_{t + 1}$) or previous
            estimation of original protein data, denoted as $\hat{x}_{0}$.
        * If the input dict of additional protein data corresponds to $x_{t + 1}$, then it should
            follow the same layout as $x_{t}$. Otherwise, it should contain following entries:
          > step: list of time-step indices (all-zeros)
          > sfea-i: initial single features of size N x L x D_si (for $x_{t + 1}$)
          > pfea-i: initial pair features of size N x L x L x D_pi (for $x_{t + 1}$)
          > sfea-u: updated single features of size N x L x D_s (for $\hat{x}_{0}$)
          > pfea-u: updated pair features of size N x L x L x D_p (for $\hat{x}_{0}$)
          > logt: residue type classification logits of size N x C x L (for $\hat{x}_{0}$)
          > cord: per-atom 3D coordinates of size N x L x M x 3 (for $\hat{x}_{0}$)
        """

        if inputs_addi is None:  # no additional inputs
            self.net['evoformer'].requires_grad_(self.training)
            outputs = self.__forward_impl(inputs, chunk_size=chunk_size)
        else:
            # build self-conditioning inputs
            if all(x == 0 for x in inputs_addi['step']):  # $\hat{x}_{0}$
                inputs_sc = {k: v.detach() for k, v in inputs_addi.items() if k != 'step'}
            elif all(x + 1 == y for x, y in zip(inputs['step'], inputs_addi['step'])):  # $x_{t + 1}$
                with torch.no_grad():

                    self.net['evoformer'].requires_grad_(False)  # no gradient computation
                    outputs = self.__forward_impl(inputs_addi)
                inputs_sc = {
                    'sfea': outputs['sfea'].detach(),
                    'pfea': outputs['pfea'].detach(),
                    'logt': outputs['1d'].permute(0, 2, 1).detach(),  # N x L x C
                    'cord': outputs['3d']['cord'][-1].detach(),
                }
            else:
                raise ValueError(f'mismatched time-steps: {inputs["step"]} / {inputs_addi["step"]}')

            # perform the forward pass w/ self-conditioning inputs
            self.net['evoformer'].requires_grad_(self.training)
            outputs = self.__forward_impl(inputs, inputs_sc=inputs_sc, chunk_size=chunk_size)

        return outputs

    def __forward_impl(self, inputs, inputs_sc=None, chunk_size=None):
        """Perform the forward pass - core implementation."""

        # build per-residue motif-or-not masks
        rmsk_vec_motf = None
        if torch.min(inputs['pmsk-ligand']) == 0:  # some residues are not perturbed
            rmsk_vec_motf = 1 - inputs['pmsk-ligand']

        # calculate various encodings
        penc_tns_ts = self.__calc_penc_tns_ts(inputs)  # positional encodings for time-steps
        penc_tns_sq = self.__calc_penc_tns_sq(inputs)  # positional encodings for sequences
        pfea_tns_rp = self.__calc_pfea_tns_rp(inputs)  # relative positional encodings
        pfea_tns_st = self.__calc_pfea_tns_st(inputs)  # structure encodings

        # initial mapping for single & pair features
        penc_tns = self.net['linear-ts'](penc_tns_ts) + self.net['linear-sq'](penc_tns_sq)
        sfea_tns_init = torch.cat([inputs['sfea-i'], penc_tns], dim=2)
        sfea_tns = self.net['linear-si'](sfea_tns_init)
        pfea_tns = self.net['linear-pi'](inputs['pfea-i'])
        pfea_tns += pfea_tns_rp + pfea_tns_st  # cooperate additional encodings

        # add ppi features
        asym_id = inputs['asym-id'][0].view(-1)
        # featurize
        if asym_id.max() == 1:
            h_len = (asym_id == 1).sum()
            a_len = (asym_id == 0).sum()
            ab_len = h_len
            chn_infos = [('H', h_len), ('L', a_len)]
        elif asym_id.max() == 2:
            h_len = (asym_id == 2).sum()
            l_len = (asym_id == 1).sum()
            ab_len = h_len + l_len
            a_len = (asym_id == 0).sum()
            chn_infos = [('H', h_len), ('L', l_len), ('A', a_len)]
        else:
            raise ValueError('Invalid asym-id')

        pfea_tns += self.crpe_encoder(chn_infos, asym_id)

        ic_feat = inputs['ic_feat'] if inputs['ic_feat'] is not None else None

        ligand_feat = {
            'seq': inputs['seq-p'][0][:ab_len],
            'cord': inputs['cord-p'][0][:ab_len, ...],
        }
        receptor_feat = {
            'seq': inputs['seq-p'][0][-a_len:],
            'cord': inputs['cord-p'][0][-a_len:, ...],
        }
        if self.use_icf and ic_feat is not None:
            # update single and pair features with inter-chain feature
            sfea_tns += self.icf_embed_sfea(ligand_feat, receptor_feat, ic_feat)
            # if self.feat_type == 'mix-3' or self.feat_type == 'mix-4':
            pfea_tns += self.icf_embed_pfea(ligand_feat, receptor_feat, ic_feat)

        pfea_tns[:, -a_len:, -a_len:, :] += self.net['struct_encoder'](inputs['a-cord'], inputs['a-cmsk'])
        # update single & pair features w/ self-conditioning inputs
        if inputs_sc is not None:
            rc_inputs = {
                'sfea': inputs_sc['sfea'] + self.net['linear-lt-sd'](inputs_sc['logt']),
                'pfea': inputs_sc['pfea'],
                'cord': inputs_sc['cord'],
            }
            mfea_tns, pfea_tns = self.net['rc_embed-sd'](
                sfea_tns.unsqueeze(dim=1), pfea_tns, rc_inputs=rc_inputs)
            sfea_tns = mfea_tns[:, 0]

        # Evoformer
        sfea_tns, pfea_tns = self.net['evoformer'](sfea_tns, pfea_tns, chunk_size=chunk_size)

        # AF2SMod
        sfea_tns_st, cord_list, param_list, plddt_list, _ = self.net['af2_smod'](
            inputs['seq-p'], sfea_tns, pfea_tns, penc_tns,
            cord_tns_init=inputs['cord-p'],
            cmsk_tns_init=inputs['cmsk-p'],
            rmsk_vec_motf=rmsk_vec_motf,
        )

        # predict denoised amino-acid sequences
        sfea_tns_st = self.net['norm_aa'](sfea_tns_st)
        logt_tns_aa = self.net['aa_pred'](sfea_tns_st)
        logt_tns_aa = logt_tns_aa.permute(0, 2, 1)  # move classification logits to the 2nd dim.

        # predict inter-residue geometries
        logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph = self.net['da_pred'](pfea_tns)

        masked = inputs['pmsk'].to(torch.bool)
        cord_masked = inputs['pmsk-ligand'].to(torch.bool)

        # pack output tensors into a dict
        outputs = {
            'sfea': sfea_tns,
            'mask': inputs['pmsk'],
            'pfea': pfea_tns,
            '1d': logt_tns_aa,
            '2d': {'cb': logt_tns_cb, 'om': logt_tns_om, 'th': logt_tns_th, 'ph': logt_tns_ph},
            '3d': {'cord': cord_list, 'param': param_list, 'plddt': plddt_list},
        }
        return outputs

    def __build_model(self):
        """Build the antibody structure prediction model."""

        # initialization
        net = nn.ModuleDict()

        # initial projection
        net['linear-ts'] = nn.Linear(self.n_dims_penc, self.n_dims_penc)
        net['linear-sq'] = nn.Linear(self.n_dims_penc, self.n_dims_penc)
        net['linear-si'] = nn.Linear(self.n_dims_sfea_init + self.n_dims_penc, self.n_dims_sfea)
        net['linear-pi'] = nn.Linear(self.n_dims_pfea_init, self.n_dims_pfea)

        net['struct_encoder'] = StructEncoder(self.n_dims_pfea)

        # Evoformer
        net['evoformer'] = EvoformerStackSS(
            num_layers=self.n_lyrs_2d,
            c_s=self.n_dims_sfea,
            c_z=self.n_dims_pfea,
        )

        # AF2SMod
        net['af2_smod'] = StructureModule(
            n_lyrs=self.n_lyrs_3d,
            n_dims_sfea=self.n_dims_sfea,
            n_dims_pfea=self.n_dims_pfea,
            n_dims_encd=self.n_dims_penc,
            pred_oxyg=self.pred_oxyg,
            pred_schn=False,
        )

        # residue type predictor
        net['norm_aa'] = nn.LayerNorm(self.n_dims_sfea)
        net['aa_pred'] = nn.Linear(self.n_dims_sfea, len(RESD_NAMES_1C))
        # net['aa_pred'] = nn.Sequential(
        #     nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
        #     nn.ReLU(),
        #     nn.Linear(self.n_dims_sfea, len(RESD_NAMES_1C))
        # )

        # PairPredictor (auxiliary predictions for inter-residue geometries)
        net['da_pred'] = PairPredictor(
            c_z=self.n_dims_pfea,
            bins=[37, 25, 25, 25],
        )

        # RcEmbedNet (for self-conditioning inputs)
        net['linear-lt-sd'] = nn.Linear(len(RESD_NAMES_1C), self.n_dims_sfea)
        net['rc_embed-sd'] = RcEmbedNet(
            n_dims_mfea=self.n_dims_sfea,
            n_dims_pfea=self.n_dims_pfea,
        )


        return net

    def __calc_penc_tns_ts(self, inputs):
        """Calculate sinusoidal positional encodings for time-steps."""

        # initialization
        n_resds = len(inputs['seq-p'][0])
        device = self.net['linear-si'].weight.device

        # calculate sinusoidal positional encodings for time-steps
        idxs_vec = torch.tensor(inputs['step'], device=device)  # N
        penc_mat = self.ts_encoder(idxs_vec)  # N x D
        penc_tns = penc_mat.unsqueeze(dim=1).repeat(1, n_resds, 1)  # N x L x D

        return penc_tns

    def __calc_penc_tns_sq(self, inputs):
        """Calculate sinusoidal positional encodings for sequences.

        Notes:
        * All the perturbed sequences are in the same length, and sinusoidal positional encodings
            only depend on the sequence length (rather the sequence itself).
        """

        # initialization
        n_smpls = len(inputs['seq-p'])
        n_resds = len(inputs['seq-p'][0])
        device = self.net['linear-si'].weight.device

        # calculate sinusoidal positional encodings for sequences
        idxs_vec = torch.arange(n_resds, device=device)  # L
        penc_mat = self.sq_encoder(idxs_vec)  # L x D
        penc_tns = penc_mat.unsqueeze(dim=0).repeat(n_smpls, 1, 1)  # N x L x D

        return penc_tns

    def __calc_pfea_tns_rp(self, inputs):
        """Calculate relative positional encodings.

        Notes:
        * All the perturbed sequences are in the same length, and relative positional encodings
            only depend on the sequence length (rather the sequence itself).
        """

        # initialization
        n_smpls = len(inputs['seq-p'])

        # calculate relative positional encodings
        pfea_tns = self.rp_encoder(inputs['seq-p'][0])  # L x L x D
        pfea_tns = pfea_tns.unsqueeze(dim=0).repeat(n_smpls, 1, 1, 1) if n_smpls > 1 else pfea_tns# N x L x L x D

        return pfea_tns

    def __calc_pfea_tns_st(self, inputs):
        """Calculate structure encodings."""

        pfea_tns = self.st_encoder(inputs['cord-p'], inputs['cmsk-p'])

        return pfea_tns
