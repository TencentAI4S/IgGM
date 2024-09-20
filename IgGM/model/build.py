# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from IgGM.utils import Registry

MODEL_REGISTRY = Registry('model')


def build_model(cfg):
    """build model by architecture name
    """
    arch = cfg.model.arch
    assert arch in MODEL_REGISTRY, f'{arch} not in model registry'
    model = MODEL_REGISTRY.get(arch)(cfg)

    return model
