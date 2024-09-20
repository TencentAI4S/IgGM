# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/6/7 14:39
import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from IgGM.config import get_config


class BaseModel(nn.Module):

    def __init__(self, activation_checkpoint_fn=None):
        super(BaseModel, self).__init__()
        self.activation_checkpoint = False
        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = checkpoint
        else:
            self.activation_checkpoint = activation_checkpoint_fn

    @classmethod
    def from_config(cls, cfg):
        raise NotImplementedError

    @classmethod
    def restore(cls, path, config=None):
        state = torch.load(path, map_location='cpu')
        config = get_config()
        config.update(state['config'])
        logging.info(config)
        model = cls(config)
        model.load_state_dict(state['model'])
        return model

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device
