# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/7 14:45
import logging
import random
import shutil
from collections import OrderedDict

import torch

from IgGM.model import BaseModel
from IgGM.protein import prot_constants as rc, ProtStruct, ProtConverter, export_fasta, PdbFixer
from IgGM.protein.data_transform import get_asym_ids
from IgGM.protein.parser import PdbParser
from IgGM.protein.prot_constants import RESD_NAMES_1C, N_ATOMS_PER_RESD
from IgGM.protein.utils import init_qta_params
from IgGM.utils import calc_trsl_vec, get_tmp_dpath


class BaseDesigner(BaseModel):
    def __init__(self):
        super().__init__()
        self.prot_struct = ProtStruct()
        self.prot_converter = ProtConverter()

    def _build_inputs(self, chains):
        """Build an input dict for model inference."""
        assert len(chains) > 0
        chain_ids = [chain["id"] for chain in chains]
        inputs = {"base": {"chain_ids": chain_ids}}

        for chain in chains:
            chain_id = chain["id"]
            sequence = chain["sequence"]
            if 'cord' not in chain:
                cord = torch.zeros(len(sequence), N_ATOMS_PER_RESD, 3)
                cmsk = torch.ones(len(sequence), N_ATOMS_PER_RESD)
            else:
                cord = chain["cord"]
                # center the coordinates
                cmsk = chain["cmsk"]
                trsl_vec = calc_trsl_vec(cord, cmsk_mat=cmsk)
                cord -= trsl_vec.view(1, 1, 3)
                cord *= cmsk.unsqueeze(dim=2)


            # valid aa sequence check
            if 'X' not in sequence:
                assert set(sequence).issubset(set(rc.RESD_NAMES_1C)), f"{chain_id} not in standard aa sequence"
            inputs[chain_id] = {"base": {"seq": sequence, "cord": cord, "cmsk": cmsk}, "feat": {}}

        # for nanobody, two chains, H and A
        if len(chains) > 1:
            complex_id = ":".join(chain_ids)
            inputs["base"]["complex_id"] = complex_id
            sequences = [chain["sequence"] for chain in chains]
            asym_id = len(chains) - get_asym_ids(sequences)
            sequences = ''.join(sequences)
            cord = [chain["cord"] if 'cord' in chain else torch.zeros(len(chain["sequence"]), N_ATOMS_PER_RESD, 3) for chain in chains]
            cord = torch.cat(cord, dim=0)
            cmsk = [chain["cmsk"] if 'cmsk' in chain else torch.ones(len(chain["sequence"]), N_ATOMS_PER_RESD) for chain in chains]
            cmsk = torch.cat(cmsk, dim=0)
            mask_ab = torch.zeros(len(sequences), dtype=torch.int8)
            mask_ab[:-len(chains[-1]["sequence"])] = 1
            if 'X' in sequences:
                mask_design = torch.LongTensor([resd == 'X' for resd in sequences])
            else:
                mask_design = torch.zeros(len(sequences), dtype=torch.int8)
            inputs[complex_id] = {
                "base": {"seq": sequences, "cord": cord, "cmsk": cmsk},
                "asym_id": asym_id.unsqueeze(dim=0),
                "mask_ab": mask_ab,
                "mask_design": mask_design,
                "a-cord": chains[-1]["cord"],
                "a-cmsk": chains[-1]["cmsk"],
                "epitope": chains[-1]["epitope"],
                "feat": {}
            }

        return inputs

    @classmethod
    def build_inputs_addi(cls, outputs):
        """Build additional inputs for the next iteration from current outputs."""

        inputs_addi = {
            'step': [0] * outputs['sfea'].shape[0],
            'sfea': outputs['sfea'].detach().clone(),
            'pfea': outputs['pfea'].detach().clone(),
            'logt': outputs['1d'].permute(0, 2, 1).detach().clone(),
            'cord': outputs['3d']['cord'][-1].detach().clone(),
        }

        return inputs_addi

    def init_prot_data(self, inputs, complex_id):
        """Randomly initialize a list of protein data dicts."""

        n_resds = len(inputs[complex_id]['base']['seq'])
        mask_ab = inputs[complex_id]['mask_ab']
        mask_design = inputs[complex_id]['mask_design']
        # randomly initialize amino-acid sequences
        aa_seqs_init = ''.join(random.choices(RESD_NAMES_1C, k=n_resds))
        aa_seq_ref = inputs[complex_id]['base']['seq']

        aa_seqs_design = [aa_seqs_init[i] if mask_design[i] else aa_seq_ref[i] for i in range(n_resds)]
        aa_seq_flat = ''.join(aa_seqs_design)

        # randomly initialize per-atom 3D coordinates
        quat_tns, trsl_tns, angl_tns = init_qta_params(1, n_resds, mode='random')
        if hasattr(self, 'scale_init') and self.scale_init:
            trsl_tns *= self.diffuser.cord_scale  # for consistency w/ <ProtDiffuser>
        param_dict = {'quat': quat_tns, 'trsl': trsl_tns, 'angl': angl_tns}
        cord_tns = self.calc_cord_from_param([aa_seq_flat], param_dict)
        cmsk_tns = ProtStruct.get_cmsk_vld(
            aa_seq_flat, cord_tns.device).view(1, n_resds, N_ATOMS_PER_RESD)

        # pack sequences & structures into dicts
        inputs[complex_id]['seq'] = aa_seq_flat
        inputs[complex_id]['cord'] = torch.where(mask_ab.view(-1, 1, 1).to(torch.bool), cord_tns[0], inputs[complex_id]['base']['cord'])
        inputs[complex_id]['cmsk'] = torch.where(mask_ab.view(-1, 1).to(torch.bool), cmsk_tns.view(-1, N_ATOMS_PER_RESD), inputs[complex_id]['base']['cmsk'])

        return inputs

    @classmethod
    def calc_coeffs(cls, alphas_bar, idx_step_curr, idx_step_next):
        """Calculate <alpha_bar> & <sigma> coefficients for DDPM sampling."""

        alpha_bar_curr = alphas_bar[idx_step_curr]
        alpha_bar_next = alphas_bar[idx_step_next]
        alpha_curr = alpha_bar_curr / alpha_bar_next
        sigma = torch.sqrt((1.0 - alpha_bar_next) / (1.0 - alpha_bar_curr) * (1.0 - alpha_curr))

        return alpha_bar_curr, alpha_bar_next, sigma


    @classmethod
    def sample_seqs_from_distr(cls, distr):
        """Sample amino-acid sequences from probabilistic distributions."""

        aa_seqs = []
        ridx_mat = distr.sample()  # N x L
        for idx_smpl in range(ridx_mat.shape[0]):
            aa_seq = ''.join([RESD_NAMES_1C[x] for x in ridx_mat[idx_smpl]])
            aa_seqs.append(aa_seq)

        return aa_seqs

    @classmethod
    def calc_cord_from_param(cls, aa_seqs, param_dict):
        """Build full-atom 3D coordinates from QTA parameters."""

        n_smpls = len(aa_seqs)
        n_resds = len(aa_seqs[0])
        aa_seq_flat = ''.join(aa_seqs)
        param_dict_flat = {k: v.view(-1, *v.shape[2:]) for k, v in param_dict.items()}
        prot_struct = ProtStruct()
        prot_converter = ProtConverter()
        prot_struct.init_from_param(
            aa_seq_flat, param_dict_flat, prot_converter, atom_set='fa')
        cord_tns = prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)

        return cord_tns

    def _get_idxs_step(self):
        """Get a list of time-step indices for sampling.

        Notes:
        * List of available schedules:
          > linear-v1: tau_{i} = round(alpha * (i - 1)) + 1
          > quad-v1: tau_{i} = round(alpha * (i - 1) ** 2) + 1
          > linear-v2: tau_{i} = round(alpha * i)
          > quad-v2: tau_{i} = round(alpha * i ** 2)
        * The main difference between v1 and v2 is that whether tau_{1} is fixed to 1.
        * Additional requirements: tau_{0} = 0, tau_{K} = T, tau_{k} >= 1 (if k != 0)
        * Selected time-step indices are sorted in the descending order.
        """

        # initialization
        n_steps_full = 200
        n_steps_smpl = self.config.steps

        # generate a list of time-step indices
        alpha = n_steps_full / n_steps_smpl
        idxs_step = [int(alpha * x + 0.5) for x in range(n_steps_smpl + 1)]

        # finalize time-step indices
        idxs_step = [max(1, min(n_steps_full, x)) for x in idxs_step]
        idxs_step[0] = 0  # ensure that tau_{0} = 0
        idxs_step[-1] = n_steps_full  # ensure that tau_{K} = T
        idxs_step.reverse()  # sort in the descending order

        return idxs_step

    @staticmethod
    def _output_to_fasta(inputs, outputs, filename):
        """export the predicted sequence to a FASTA file"""
        complex_id = inputs["base"]["complex_id"]
        complex_ids = complex_id.split(':')
        start = 0
        sequences = []
        ids = []

        for chn_id in complex_ids:
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs['seq'][start:start + len(aa_seq)]
            sequences.append(aa_seq)
            ids.append(chn_id)
            start += len(aa_seq)

        export_fasta(sequences, ids=ids, output=filename)

    @staticmethod
    def _output_to_pdb(inputs, outputs, filename):
        """Build a dict of protein structure data."""
        pred_info = 'REMARK 250 Structure predicted by IgGM\n'
        ligand_id = inputs["base"]["ligand_id"]
        receptor_id = inputs["base"]["receptor_id"]

        tmp_path = get_tmp_dpath()
        prot_data = OrderedDict()
        start = 0
        pdb_fixer = PdbFixer()
        for chn_id in ligand_id.split(":"):
            prot_data_chain = OrderedDict()
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs['seq'][start:start + len(aa_seq)]
                pred_info += f'REMARK 250 Predicted Sequence for chain {chn_id}: {aa_seq}\n'
            prot_data_chain[chn_id] = {
                'seq': aa_seq,
                'cord': outputs['cord'][start:start + len(aa_seq)],
                'cmsk': outputs['cmsk'][start:start + len(aa_seq)],
            }
            chain_save_path = f'{tmp_path}/{chn_id}.pdb'
            chain_add_path = f'{tmp_path}/{chn_id}_add.pdb'
            PdbParser.save_multimer(prot_data_chain, chain_save_path, pred_info=pred_info)
            pdb_fixer.add_atoms_api(chain_save_path, chain_add_path)
            chn_id_data = PdbParser.load(chain_add_path, aa_seq=aa_seq)
            prot_data[chn_id] = {
                'seq': chn_id_data[0],
                'cord': chn_id_data[1],
                'cmsk': chn_id_data[2],
            }
            start += len(aa_seq)

        for chn_id in receptor_id.split(":"):
            aa_seq = inputs[chn_id]['base']['seq']
            prot_data[chn_id] = {
                'seq': aa_seq,
                'cord': inputs[chn_id]['base']['cord'],
                'cmsk': inputs[chn_id]['base']['cmsk'],
            }

        PdbParser.save_multimer(prot_data, filename, pred_info=pred_info)
        # remove the tmp directory tmp_path
        shutil.rmtree(tmp_path)
        logging.info(f'PDB file generated: {filename}')


    def calc_cords_from_param(self, aa_seqs, param_dict):
        """Build full-atom 3D coordinates from QTA parameters."""

        n_smpls = len(aa_seqs)
        n_resds = len(aa_seqs[0])
        aa_seq_flat = ''.join(aa_seqs)
        param_dict_flat = {k: v.view(-1, *v.shape[2:]) for k, v in param_dict.items()}
        self.prot_struct.init_from_param(
            aa_seq_flat, param_dict_flat, self.prot_converter, atom_set='fa')
        cord_tns = self.prot_struct.cord_tns.view(n_smpls, n_resds, N_ATOMS_PER_RESD, 3)

        return cord_tns
