# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from termcolor import colored


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop('root_name') + '.'
        self._abbrev_name = kwargs.pop('abbrev_name', '')
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + '.'
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored('WARNING', 'red', attrs=['blink'])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored('ERROR', 'red', attrs=['blink', 'underline'])
        else:
            return log
        return prefix + ' ' + log


def setup_logger(name=None, color: bool = True, rank=0, abbrev_name=None):
    """Initialize default logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    abbrev_name = abbrev_name or name
    plain_formatter = logging.Formatter(
        '[%(asctime)-15s %(levelname)s %(filename)s:L%(lineno)d] %(message)s', datefmt='%m/%d %H:%M:%S'
    )
    if rank == 0:
        if color:
            formatter = _ColorfulFormatter(
                colored('[%(asctime)s %(name)s]: ', 'green') + '%(message)s',
                datefmt='%m/%d %H:%M:%S',
                root_name=str(name),
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        if len(logger.handlers) > 0:
            for h in logger.handlers:
                if isinstance(h, logging.StreamHandler):
                    h.setLevel(logging.DEBUG)
                    h.setFormatter(formatter)
        else:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(logging.DEBUG)
            ch.setFormatter(formatter)
            logger.addHandler(ch)


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() +
                int(datetime.now().strftime('%S%f')) +
                int.from_bytes(os.urandom(2), 'big')
                )
        logger = logging.getLogger(__name__)
        logger.info('Using a generated random seed {}'.format(seed))

    if seed is not None and seed > 0:
        np.random.seed(seed)
        torch.set_rng_state(torch.manual_seed(seed).get_state())
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def setup(inference=False, seed=42):
    if inference:
        torch.set_grad_enabled(False)

    seed_all_rng(seed)
    setup_logger()
