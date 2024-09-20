# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from .build import build_model
# auto register model here
from .arch import PPIModel, DesignModel, BaseModel
from .pretrain import esm_ppi_650m_ab, antibody_design_trunk, IGSO3Buffer_trunk
