# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from ml_collections import FieldReference

from IgGM.utils import Registry
from .config_node import CfgNode as CN

CONFIG_REGISTRY = Registry('config')

_C = CN()
_C.model = CN()
# default model
_C.model.arch = 'IgGM_ab'
_C.model.attention_mode = "native"


@CONFIG_REGISTRY.register('esm')
def llm_model(_C):
    _C.model.arch = 'PPIModel'
    _C.model.tokenizer = 'ESM-1b'
    _C.model.embedding_dim = 1280
    _C.model.num_layers = 33
    _C.model.num_heads = 20
    _C.model.token_dropout = True
    _C.model.use_crp_embeddings = True
    return _C


@CONFIG_REGISTRY.register('IgGM_ab')
def complex_structure_model(_C):
    _C.model.arch = 'ComplexStructureModel'
    _C.model.c_s = 192
    _C.model.c_z = 128
    _C.model.init_c_s = 1280
    _C.model.init_c_z = 660
    _C.model.position_embedding_dim = 64
    _C.model.use_residue_embedding = True
    _C.model.tm_enabled = True
    _C.model.num_2d_layers = 16
    _C.model.num_3d_layers = 8
    _C.model.num_recycles = 1
    _C.model.num_3d_recycles = 8
    return _C


@CONFIG_REGISTRY.register()
def IgGM_ag(_C):
    _C.model.arch = 'IgGMAg'
    _C.model.plm = CN()
    c_s = FieldReference(192)
    c_z = FieldReference(128)
    init_c_s = FieldReference(1280)
    init_c_z = FieldReference(660)
    _C.model.plm.c_s = init_c_s
    _C.model.plm.c_z = init_c_z
    _C.model.ligand = CN()
    _C.model.ligand.c_s = c_s
    _C.model.ligand.c_z = c_z
    _C.model.ligand.init_c_s = init_c_s
    _C.model.ligand.init_c_z = init_c_z

    _C.model.ligand.position_embedding_dim = 64
    _C.model.ligand.use_residue_embedding = False
    _C.model.ligand.tm_enabled = True

    _C.model.ligand.num_2d_layers = 16
    _C.model.ligand.num_3d_layers = 8

    _C.model.ligand.num_recycles = 1
    _C.model.ligand.num_3d_recycles = 8

    _C.model.docking = CN()
    _C.model.docking.ligand_c_s = c_s
    _C.model.docking.ligand_c_z = c_z
    _C.model.docking.receptor_c_s = 256
    _C.model.docking.receptor_c_z = 128
    _C.model.docking.use_icf = False

    _C.model.docking.num_2d_layers = 32
    _C.model.docking.num_3d_layers = 8

    _C.model.docking.num_recycles = 1
    _C.model.docking.num_3d_recycles = 8
    return _C


@CONFIG_REGISTRY.register()
def IgGM_tcr(_C):
    _C.model.arch = 'IgGMTCR'
    _C.model.plm = CN()
    c_s = FieldReference(192)
    c_z = FieldReference(128)
    init_c_s = FieldReference(1280)
    init_c_z = FieldReference(660)

    _C.model.plm.c_s = init_c_s
    _C.model.plm.c_z = init_c_z

    _C.model.ligand = CN()
    _C.model.ligand.c_s = c_s
    _C.model.ligand.c_z = c_z
    _C.model.ligand.init_c_s = init_c_s
    _C.model.ligand.init_c_z = init_c_z
    _C.model.ligand.position_embedding_dim = 64
    _C.model.ligand.use_residue_embedding = False
    _C.model.ligand.tm_enabled = True
    _C.model.ligand.num_2d_layers = 16
    _C.model.ligand.num_3d_layers = 8
    _C.model.ligand.num_recycles = 1
    _C.model.ligand.num_3d_recycles = 8

    _C.model.receptor = CN()
    _C.model.receptor.c_s = c_s
    _C.model.receptor.c_z = c_z
    _C.model.receptor.init_c_s = init_c_s
    _C.model.receptor.init_c_z = init_c_z
    _C.model.receptor.position_embedding_dim = 64
    _C.model.receptor.use_residue_embedding = False
    _C.model.receptor.tm_enabled = True
    _C.model.receptor.num_2d_layers = 16
    _C.model.receptor.num_3d_layers = 8
    _C.model.receptor.num_recycles = 1
    _C.model.receptor.num_3d_recycles = 8

    _C.model.docking = CN()
    _C.model.docking.ligand_c_s = c_s
    _C.model.docking.ligand_c_z = c_z
    _C.model.docking.receptor_c_s = c_s
    _C.model.docking.receptor_c_z = c_z
    _C.model.docking.coord_init = "tcr"
    _C.model.docking.num_2d_layers = 16
    _C.model.docking.num_3d_layers = 8

    _C.model.docking.num_recycles = 1
    _C.model.docking.num_3d_recycles = 8
    return _C
