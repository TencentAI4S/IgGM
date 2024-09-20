# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import hashlib
import logging
import random

from contextlib import contextmanager
from datetime import datetime


def get_rand_str():
    """Get a randomized string.
    Returns:
        rand_str: randomized string
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = f'{timestamp}_{rand_val}'
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use.
        This would only need to be changed if a custom level greater than CRITICAL
        is defined.
    """
    # https://gist.github.com/simon-weber/7853144

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
