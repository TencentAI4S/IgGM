# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
import logging
import sys
from functools import partial

__all__ = ['get_registry', 'Registry']

GLOBAL_REGISTRY = {}
logger = logging.getLogger(__file__)


def _register_generic(module_dict, module_name, module):
    if module_name in module_dict:
        logger.warning(f'exist moudle: {module_name}, override it with new moudle: {module.__name__}')

    module_dict[module_name] = module


def get_registry(name):
    reg = GLOBAL_REGISTRY.get(name, None)
    if reg is None:
        print(f'not registry named: {name}', file=sys.stderr)

    return reg


class Registry(dict):
    """
    registry helper，
    Eg. creeting a registry:
        some_registry = Registry({'default': default_module})

    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register('foo_module', foo)

    2): used as decorator when declaring the module:
        @some_registry.register('foo_module')
        @some_registry.register('foo_modeul_nickname')
        def foo():
            ...

    get register moudule, eg:
        f = some_registry['foo_modeul']
    """

    def __init__(self, name='registry', *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)
        self.name = name
        GLOBAL_REGISTRY[name] = self

    @classmethod
    def get_registry(cls, name):
        reg = GLOBAL_REGISTRY.get(name, None)
        if reg is None:
            print(f'not registry named: {name}', file=sys.stderr)

        return reg

    def register(self, module_name=None, module=None, *args, **kwargs):
        """register decorator
        Args:
            module_name: str
            module： function
        """

        # register function
        if module is not None:
            if len(kwargs):
                module = partial(module, *args, **kwargs)
            _register_generic(self, module_name, module)

            return

        # register class
        def register_fn(fn):
            name = module_name if module_name else fn.__name__
            if len(kwargs):
                fn = partial(fn, *args, **kwargs)
            _register_generic(self, name, fn)

            return fn

        return register_fn
