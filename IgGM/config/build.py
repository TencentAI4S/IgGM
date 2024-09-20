# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from .default import _C, CONFIG_REGISTRY


def get_config(name=None, clone: bool = False):
    if name is None:
        cfg = _C
    else:
        cfg = CONFIG_REGISTRY[name](_C)

    if clone:
        return cfg.clone()

    return cfg
