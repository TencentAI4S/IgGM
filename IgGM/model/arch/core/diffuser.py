"""Protein diffusion model for amino-acid sequences & backbone structures.

Notes:
* For <RcsbMonoDataset>, it is guaranteed (by construction) that there is no non-standard residue
    types in the amino-acid sequence.
"""

import logging
import random

import numpy as np
import torch
from torch import nn

from IgGM.protein import AtomMapper
from IgGM.protein.prot_constants import RESD_NAMES_1C
from IgGM.utils import ss2ptr, so3_scale, ptr2ss, IsotropicGaussianSO3


class Diffuser:
    """Protein diffusion model for amino-acid sequences & backbone structures."""

    def __init__(
            self,
            n_steps=200,  # number of time steps in the diffusion process
            pert_seq=True,  # whether to perturb amino-acid sequences
            cord_scale=4.0,  # coordinate scaling factor (in Angstrom)
            igso3_buffer=None,  # buffered rotational matrices sampled from IGSO(3) distributions
    ):
        """Constructor function."""

        # setup configurations
        self.n_steps = n_steps
        self.pert_seq = pert_seq
        self.cord_scale = cord_scale
        self.igso3_buffer = igso3_buffer

        # additional configurations
        self.rota_buf_size = 1024  # number of rotation matrices buffered for each IGSO(3) distr.
        self.atom_mapper = AtomMapper()
        self.resd_names = RESD_NAMES_1C  # 20 standard AA type tokens
        self.n_tokns = len(self.resd_names)
        self.mask_rate = None

        # initialize variance schedules
        self.seq_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)
        self.trsl_schedule = LinearSchedule(n_steps=self.n_steps, beta_min=0.01, beta_max=0.07)
        self.rota_schedule = CosineSchedule(n_steps=self.n_steps, offset=0.008, beta_max=0.999)

        # prepare transition matrices for sequence perturbation
        self.__build_trmat_list()

        # prepare IGSO(3) distributions for rotation perturbation
        self.__build_igso3_list()

    def run(self, prot_data_orig, idxs_step=None, return_time_steps=False):
        """Run the protein diffusion model.

        Args:
        * prot_data_orig: original protein data dict
        * idxs_step: (optional) list of time steps (ranging from 1 to T)
        * pert_seq: (optional) whether to perturb amino-acid sequences
        * pert_trsl: (optional) whether to perturb translational components of backbone structures
        * pert_rota: (optional) whether to perturb rotational components of backbone structures
        * pmsk_vec: (optional) per-residue perturb-or-not masks of size L
        * pmsk_vec_ligand: (optional) per-residue perturb-or-not masks for ligand

        Returns:
        * prot_data_pert: perturbed protein data dict

        Notes:
        * For training w/ self-conditioning inputs, it is required that the time-step ranges from 2
            to T, instead of the default range (from 1 to T).
        """

        # initialization
        n_resds = len(prot_data_orig['seq'])
        device = prot_data_orig['cord'].device

        # obtain the original sequence & structure
        aa_seq_orig = prot_data_orig['seq']
        cord_tns_orig = prot_data_orig['cord']
        cmsk_mat_orig = prot_data_orig['cmsk']
        pmsk_vec = prot_data_orig['mask_design']
        pmsk_vec_ligand = prot_data_orig['mask_ab']


        # convert the original sequence & structure into Prob-Trsl-Rota parameters
        prob_tns_orig, trsl_tns_orig, rota_tns_orig, fmsk_mat_orig = \
            ss2ptr([aa_seq_orig], cord_tns_orig, cmsk_mat_orig)

        # perturb probabilistic distributions
        trmat_ac = self.trmat_list_ac[idxs_step].to(device)
        prob_tns_pert = torch.matmul(prob_tns_orig, trmat_ac)
        prob_tns_pert = nn.functional.normalize(prob_tns_pert, p=1.0, dim=2)
        prob_tns_pert = torch.where(
            pmsk_vec.view(-1 ,n_resds, 1).to(torch.bool), prob_tns_pert, prob_tns_orig)

        # perturb translation vectors
        alpha_bar = self.trsl_schedule.alphas_bar[idxs_step].to(device)
        trsl_tns_nois = torch.randn_like(trsl_tns_orig[0])
        trsl_tns_pert = torch.sqrt(alpha_bar) * trsl_tns_orig[0] + \
                                  self.cord_scale * torch.sqrt(1.0 - alpha_bar) * trsl_tns_nois
        trsl_tns_pert = torch.where(
            pmsk_vec_ligand.view(n_resds, 1).to(torch.bool), trsl_tns_pert, trsl_tns_orig)

        # perturb rotation matrices
        alpha_bar = self.rota_schedule.alphas_bar[idxs_step].to(device)
        rota_buf = self.rota_buf_list_fwd[idxs_step].to(device)
        idxs_buf = random.choices(range(self.rota_buf_size), k=n_resds)
        rota_tns_nois = rota_buf[idxs_buf]
        rota_tns_pert= torch.bmm(
            so3_scale(rota_tns_orig[0], torch.sqrt(alpha_bar)), rota_tns_nois)
        rota_tns_pert = torch.where(
            pmsk_vec_ligand.view(n_resds, 1, 1).to(torch.bool), rota_tns_pert, rota_tns_orig)

        # convert Prob-Trsl-Rota parameters into amino-acid sequences & per-atom 3D coordinates
        aa_seqs_pert, cord_tns_pert, cmsk_tns_pert = \
            ptr2ss(prob_tns_pert, trsl_tns_pert, rota_tns_pert, fmsk_mat_orig, stoc_seq=True)

        # pack perturbed amino-acid sequences & backbone structures into a dict
        prot_data_pert = {
            'step': [idxs_step],
            'seq-o': aa_seq_orig,
            'cord-o': cord_tns_orig,  # L x M x 3
            'cmsk-o': cmsk_mat_orig,  # L x M
            'pmsk': pmsk_vec,  # L
            'pmsk-ligand': pmsk_vec_ligand,  # L
            'seq-p': aa_seqs_pert,
            'cord-p': cord_tns_pert,  # N x L x M x 3
            'cmsk-p': cmsk_tns_pert,  # N x L x M

            'asym-id': prot_data_orig['asym_id'].detach().clone(),
            'a-cord': prot_data_orig['a-cord'].detach().clone(),
            'a-cmsk': prot_data_orig['a-cmsk'].detach().clone(),
        }

        if return_time_steps:
            return prot_data_pert, idxs_step
        return prot_data_pert

    def __build_trmat_list(self):
        """Build a list of transition matrices."""

        # initialize basic transition matrices
        trmat_diag = torch.eye(self.n_tokns)
        trmat_unif = torch.ones((self.n_tokns, self.n_tokns)) / self.n_tokns

        # build a list of transition matrices (single step & accumulated)
        self.trmat_list_st = []  # single-step (Q_t)
        self.trmat_list_ac = []  # accumulated (\bar{Q}_t = Q_1 * Q_2 * ... * Q_t)
        for idx_step, beta in enumerate(self.seq_schedule.betas):
            if idx_step == 0:
                trmat_st = trmat_diag
                trmat_ac_prev = trmat_diag
            else:
                trmat_st = (1 - beta) * trmat_diag + beta * trmat_unif
                trmat_ac_prev = self.trmat_list_ac[-1]
            trmat_ac = torch.matmul(trmat_ac_prev, trmat_st)
            self.trmat_list_st.append(trmat_st)
            self.trmat_list_ac.append(trmat_ac)

    def __build_igso3_list(self):
        """Build a list of IGSO(3) distributions."""

        # build IGSO(3) distributions for the forward process (x_{0} -> x_{t})
        logging.info('building forward IGSO(3) distributions ...')
        self.rota_buf_list_fwd = [None]  # skip the first entry
        for idx, stdev in enumerate(self.rota_schedule.sigmas[1:]):
            if self.igso3_buffer is None:
                igso3 = IsotropicGaussianSO3(eps=stdev.view(1))
                rota_buf = igso3.sample_batch(torch.Size([self.rota_buf_size]))[:, 0]
            else:
                rota_buf = self.igso3_buffer.sample(stdev.item(), self.rota_buf_size)
            self.rota_buf_list_fwd.append(rota_buf)

        # build IGSO(3) distributions for the backward process (x_{0} & x_{t} -> x_{t-1})
        logging.info('building backward IGSO(3) distributions ...')
        self.rota_buf_list_bwd = [None, None]  # skip the first two entries
        for idx, stdev in enumerate(self.rota_schedule.betas_tld[2:]):
            if self.igso3_buffer is None:
                igso3 = IsotropicGaussianSO3(eps=stdev.view(1))
                rota_buf = igso3.sample_batch(torch.Size([self.rota_buf_size]))[:, 0]
            else:
                rota_buf = self.igso3_buffer.sample(stdev.item(), self.rota_buf_size)
            self.rota_buf_list_bwd.append(rota_buf)

class VarianceSchedule():
    """General variance schedule for DDPM training & sampling.

    Notes:
    > alpha_{t} = 1 - beta_{t}
    > alpha_bar_{t} = alpha_{1} * alpha_{2} * ... * alpha_{t}

    Requirements:
    > beta_{0} = 0 (which leads to alpha_{0} = 1 and alpha_bar_{0} = 1)
    > beta_{t} should be monotonically increasing
    > alpha_bar_{1} should be close to 1
    > alpha_bar_{T} should be close to 0
    """

    def __init__(self):
        """Constructor function."""

        self.n_steps = None  # integer; number of diffusion steps (T)
        self.betas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas = None  # 1D array of length <T+1> (from t=0 to t=T)
        self.alphas_bar = None  # 1D array of length <T+1> (from t=0 to t=T)

    def calc_vars(self):
        """Calculate variance coefficients for forward & backward processes."""

        self.sigmas = torch.sqrt(1.0 - self.alphas_bar)
        self.betas_tld = torch.sqrt(
            self.betas[1:] * (1.0 - self.alphas_bar[:-1]) / (1.0 - self.alphas_bar[1:]))
        self.betas_tld = nn.functional.pad(self.betas_tld, (1, 0), mode='constant', value=0.0)

    def sample(self, idxs_step):
        """Build a variance schedule w/ sub-sampled time-steps to match marginal distributions."""

        assert (min(idxs_step) >= 1) and (max(idxs_step) <= self.n_steps)

        obj = VarianceSchedule()
        obj.n_steps = len(idxs_step)
        obj.alphas_bar = self.alphas_bar[[0] + sorted(idxs_step)]
        obj.alphas = torch.ones_like(obj.alphas_bar)  # t=0 corresponds to no perturbation
        obj.alphas[1:] = obj.alphas_bar[1:] / obj.alphas_bar[:-1]
        obj.betas = 1.0 - obj.alphas

        return obj

class LinearSchedule(VarianceSchedule):
    """Linear variance schedule (as proposed in DDPM)."""

    def __init__(self, n_steps=1000, beta_min=0.0001, beta_max=0.02):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        # additional configurations
        self.betas = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)
        super().calc_vars()

class CosineSchedule(VarianceSchedule):
    """Cosine variance schedule (as proposed in Improved DDPM)."""

    def __init__(self, n_steps=4000, offset=0.008, beta_max=0.999):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_steps = n_steps
        self.offset = offset
        self.beta_max = beta_max  # to prevent singularities at the end of diffusion process

        # additional configurations
        t_vals = torch.arange(self.n_steps + 1) / self.n_steps
        f_vals = torch.cos((t_vals + offset) / (1 + offset) * np.pi / 2) ** 2
        self.betas = torch.clamp(1 - f_vals[1:] / f_vals[:-1], min=0.0, max=self.beta_max)
        self.betas = nn.functional.pad(self.betas, (1, 0), mode='constant', value=0.0)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, 0)  # re-calculated for consistency
        super().calc_vars()
