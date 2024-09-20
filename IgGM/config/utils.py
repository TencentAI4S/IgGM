# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import functools
import inspect

from .config_node import CfgNode
from ml_collections import ConfigDict

def configurable(init_func):
    """
    Decorate a class's __init__ method so that it can be called with a CfgNode
    object using the class's from_config classmethod.

    Examples:
    ::
        class A:
            @configurable
            def __init__(self, a, b=2, c=3):
                pass

            @classmethod
            def from_config(cls, cfg):
                # Returns kwargs to be passed to __init__
                return {'a': cfg.A, 'b': cfg.B}

        a1 = A(a=1, b=2)  # regular construction
        a2 = A(cfg)       # construct with a cfg
        a3 = A(cfg, b=3, c=4)  # construct with extra overwrite
    """
    assert init_func.__name__ == '__init__', '@configurable should only be used for __init__!'

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        try:
            from_config_func = type(self).from_config
        except AttributeError:
            raise AttributeError('Class with @configurable must have a "from_config" classmethod.')
        if not inspect.ismethod(from_config_func):
            raise TypeError('Class with @configurable must have a "from_config" classmethod.')

        if _called_with_cfg(*args, **kwargs):
            explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
            init_func(self, **explicit_args)
        else:
            init_func(self, *args, **kwargs)

    return wrapped


def _get_args_from_config(from_config_func, *args, **kwargs):
    """
    Use `from_config` to obtain explicit arguments.

    Returns:
        dict: arguments to be used for cls.__init__
    """
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != 'cfg':
        raise TypeError(f'{from_config_func.__self__}.from_config must take "cfg" as the first argument!')
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD] for param in signature.parameters.values()
    )
    if support_var_arg:  # forward all arguments to from_config, if from_config accepts them
        ret = from_config_func(*args, **kwargs)
    else:
        # forward supported arguments to from_config
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        # forward the other arguments to __init__
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    """
    Returns:
        bool: whether the arguments contain CfgNode and should be considered
            forwarded to from_config.
    """
    if len(args) and isinstance(args[0], (CfgNode, ConfigDict)):
        return True

    if isinstance(kwargs.pop('cfg', None), (CfgNode, ConfigDict)):
        return True
    # `from_config`'s first argument is forced to be 'cfg'.
    # So the above check covers all cases.
    return False
