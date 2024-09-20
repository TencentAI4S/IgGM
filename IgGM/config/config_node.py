# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import copy
import logging
from typing import List
from typing import Union, IO, Dict, Any

import yaml
from ml_collections import ConfigDict

from IgGM.utils import jload


class CfgNode(ConfigDict):
    """compact yacs config node api"""

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        """
        Defines how a config file is opened. May be overridden to support
        different file schemas.
        """
        return open(filename, 'r')

    def freeze(self):
        return self.lock()

    def defrost(self):
        return self.unlock()

    def clone(self):
        return copy.deepcopy(self)

    def merge_from_other_cfg(self, cfg_other: 'CfgNode'):
        return self.update(cfg_other)

    def update(self, cfg_other):
        if isinstance(cfg_other, dict):
            cfg_other = CfgNode(cfg_other)
        super().update(cfg_other)

    def merge_from_list(self, cfg_list: List[str]):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this CfgNode. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        assert len(cfg_list) % 2 == 0, f'Override list has odd length: {cfg_list}; it must be a list of pairs'
        self.update_from_flattened_dict({k: v for (k, v) in cfg_list})

    @classmethod
    def load_yaml_with_base(cls, filename: str, allow_unsafe: bool = False) -> Dict[str, Any]:
        """
        Just like `yaml.load(open(filename))`, but inherit attributes from its
            `_BASE_`.

        Args:
            filename (str or file-like object): the file name or file of the current config.
                Will be used to find the base config file.
            allow_unsafe (bool): whether to allow loading the config file with
                `yaml.unsafe_load`.

        Returns:
            (dict): the loaded yaml
        """
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    'Loading config {} with yaml.unsafe_load. Your machine may '
                    'be at risk if the file contains malicious content.'.format(filename)
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        """
        Merge configs from a given yaml file.

        Args:
            cfg_filename: the file name of the yaml config.
            allow_unsafe: whether to allow loading the config file with
                `yaml.unsafe_load`.
        """
        if cfg_filename.endswith('json'):
            loaded_cfg = jload(cfg_filename)
        else:
            loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)

        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    def dump(self, **kwargs):
        return self.to_yaml(**kwargs)

    def is_frozen(self):
        return self.is_locked


CN = CfgNode
