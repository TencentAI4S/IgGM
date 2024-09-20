# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging
import time

import torch
from torch import nn
from torch.distributions import Categorical

from IgGM.model import DesignModel, PPIModel
from IgGM.protein import ProtStruct
from IgGM.protein.data_transform import get_asym_ids
from IgGM.utils import to_device, IGSO3Buffer, replace_with_mask
from .base_designer import BaseDesigner
from ..model.arch.core.diffuser import Diffuser


class AbDesigner(BaseDesigner):
    """The antibody & antigen multimer structure predictor.
    """

    def __init__(self, ppi_path, design_path, buffer_path, config):
        super().__init__()
        logging.info('restoring the pre-trained IgGM-PPI-SeqPT model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        config.c_s = self.plm_featurizer.c_s
        config.c_p = self.plm_featurizer.c_z
        self.config = config
        self.igso3_buffer = IGSO3Buffer()
        self.igso3_buffer.load(buffer_path)
        self.diffuser = Diffuser(igso3_buffer=self.igso3_buffer)
        self.buffer_path = buffer_path
        logging.info('restoring the pre-trained IgGM design model ...')
        self.model = DesignModel.restore(design_path, config)
        self.idxs_step = self._get_idxs_step()
        self.eval()

    def _build_inputs(self, chains):
        """Build an input dict for model inference."""
        num_chains = len(chains)
        inputs = super()._build_inputs(chains)
        if num_chains == 2:  # nanobody, two chains, H and A
            ligand_id = 'H'
        else:  # antibody, three chains, H, L, A
            ligand_id = ':'.join(['H', 'L'])  # pseudo chain ID for the VH-VL complex
            h_seq = inputs["H"]["base"]["seq"]
            l_seq = inputs["L"]["base"]["seq"]
            inputs[ligand_id] = {
                'base': {'seq': h_seq + l_seq},
                'asym_id': get_asym_ids([h_seq, l_seq]).unsqueeze(dim=0),
                'feat': {},
            }

        inputs["base"]["ligand_id"] = ligand_id
        inputs["base"]["receptor_id"] = "A"

        complex_id = ':'.join([ligand_id, 'A'])  # H:A or H:L:A
        prot_data = self.init_prot_data(inputs, complex_id) # initialize the protein data from sample noise

        prot_data["base"]["complex_id"] = complex_id

        epitope = prot_data[complex_id]['epitope']

        if epitope is None:
            logging.info('no epitope information provided, the position placement will be determined by the model')

        return prot_data

    def forward(self, inputs, chunk_size=None):
        """Run the antibody & antigen multimer structure predictor.
        """
        start = time.time()
        inputs = to_device(inputs, device=self.device)
        complex_id = inputs["base"]["complex_id"]
        idxs_step = self.idxs_step[:-1]  # skip tau_{0} (which is fixed to 0)
        # update sequences & structures through multiple iterations
        prot_data_curr = inputs[complex_id]  # record protein data in the sampling process

        inputs_addi = None  # will be initialized at the end of first iterations

        for idx_step in idxs_step:
            # initialization  # use sequences & structures from the last iter.
            aa_seqs_pred, cord_tns_pred, inputs_addi = \
                self.__sample_cm_ss2ss(prot_data_curr, idx_step, inputs_addi, chunk_size=chunk_size)

            prot_data_curr['seq'] = aa_seqs_pred
            prot_data_curr['cord'] = cord_tns_pred
            prot_data_curr['cmsk'] = ProtStruct.get_cmsk_vld(aa_seqs_pred, self.device)


        logging.info('start ab design model in %.2f second', time.time() - start)

        return prot_data_curr

    @torch.no_grad()
    def infer(self, chains, *args, **kwargs):
        assert all(x["id"] in {'H', 'L', 'A'} for x in chains), 'chain ID must be "H", "L" or "A"'
        assert len(chains) in (2, 3), f'FASTA file should contain 2 or 3 chains'

        inputs = self._build_inputs(chains)
        outputs = self.forward(inputs, *args, **kwargs)
        return inputs, outputs

    def infer_pdb(self, chains, filename, *args, **kwargs):
        inputs, outputs = self.infer(chains, *args, **kwargs)
        complex_id = inputs["base"]["complex_id"]
        raw_seqs = {}
        for chain_id in complex_id.split(":"):
            raw_seq = inputs[chain_id]["base"]["seq"]
            raw_seqs[chain_id] = raw_seq

        self._output_to_fasta(inputs, outputs, filename[:-4] + ".fasta")
        self._output_to_pdb(inputs, outputs, filename)


    def __sample_cm_ss2ss(self, prot_data_curr, idx_step, inputs_addi, chunk_size=None):
        """Sample amino-acid sequences & backbone structures w/ CM."""

        # perform the forward pass
        inputs = self.__build_inputs_cm(prot_data_curr, idx_step)
        outputs = self.model(inputs, inputs_addi=inputs_addi, chunk_size=chunk_size)

        inputs_addi = self.build_inputs_addi(outputs)  # for the next iteration

        # build predicted sequences & structures
        prob_tns = nn.functional.softmax(outputs['1d'].permute(0, 2, 1), dim=2)
        distr = Categorical(probs=prob_tns)
        aa_seqs_pred = self.sample_seqs_from_distr(distr)
        pmsk_vec = inputs['pmsk']
        aa_seqs_pred = [replace_with_mask(inputs['seq-o'], aa_seq_pred, pmsk_vec) for aa_seq_pred in aa_seqs_pred]
        cord_tns_pred = self.calc_cords_from_param(aa_seqs_pred, outputs['3d']['param'][-1])
        pmsk_vec_ligand = inputs['pmsk-ligand']
        cord_tns_pred = torch.where(pmsk_vec_ligand.view(1, -1, 1, 1).to(torch.bool), cord_tns_pred, inputs['cord-o'])

        return aa_seqs_pred[0], cord_tns_pred[0], inputs_addi

    def __build_inputs_cm(self, prot_data_curr, idx_step):
        """Build inputs for CM-based sampling."""

        # initialization
        # randomly perturb amino-acid sequences and/or backbone structures
        prot_data_pert = self.diffuser.run(prot_data_curr, idx_step)
        # build a dict of input tensors
        inputs = self.model.featurize(self.plm_featurizer, prot_data_pert)
        ic_feat = torch.zeros_like(prot_data_curr['asym_id'])
        ag_len = len(prot_data_curr['epitope'])
        ic_feat[:, -ag_len:] = prot_data_curr['epitope']

        inputs['ic_feat'] = ic_feat.unsqueeze(-1).type_as(inputs['sfea-i'])

        return inputs

