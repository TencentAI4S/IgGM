# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import os
import urllib

import torch

def esm_ppi_650m_ab():
    print("Loading ESM PPI 650m AB model")
    return load_model_hub("esm_ppi_650m_ab")



def antibody_design_trunk():
    print("Loading Antibody Design Model")
    return load_model_hub("antibody_design_trunk")

def IGSO3Buffer_trunk():
    print("Download IGSO3Buffer for accelerating SO3 operations")
    return load_model_hub("igso3_buffer")


def load_model_hub(model_name):
    model_path = _download_model_data(model_name)
    return model_path


def _download_model_data(model_name):
    url = f"https://zenodo.org/records/13253983/files/{model_name}.pth?download=1"
    model_path = load_hub_workaround(url, model_name)
    return model_path


def load_hub_workaround(url, model_name):
    try:
        os.makedirs(f"{os.getcwd()}/checkpoints", exist_ok=True)
        model_path = f"{os.getcwd()}/checkpoints/{model_name}.pth"
        if os.path.exists(model_path):
            return model_path
        torch.hub.download_url_to_file(url, progress=True, dst=model_path)
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return model_path
