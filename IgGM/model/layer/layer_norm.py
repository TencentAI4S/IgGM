# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2023/7/31 15:32
import logging
import numbers
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from IgGM.utils.tensor import make_viewless_tensor

_shape_t = Union[int, List[int], torch.Size]

try:
    from apex.normalization.fused_layer_norm import fused_layer_norm, fused_layer_norm_affine
    from apex.contrib.layer_norm.layer_norm import _fast_layer_norm as fast_layer_norm

    logging.warning("using apex fused layer norm")

except ImportError:
    fused_layer_norm = None
    fused_layer_norm_affine = None
    fast_layer_norm = None


def _set_sequence_parallel_enabled(
        param: torch.Tensor,
        sequence_parallel_enabled: bool,
) -> None:
    setattr(param, "sequence_parallel_enabled", sequence_parallel_enabled)


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. See description of nn.LayerNorm"""
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self,
                 normalized_shape: _shape_t,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 sequence_parallel_enabled: bool = False,
                 persist_layer_norm: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        persist_ln_hidden_sizes = (1024, 1536, 2048, 2304, 3072, 3840, 4096,
                                   5120, 6144, 8192, 10240, 12288, 12800, 15360,
                                   16384, 18432, 20480, 24576, 25600, 30720, 32768,
                                   40960, 49152, 65536)
        if normalized_shape not in persist_ln_hidden_sizes or (fast_layer_norm is not None):
            persist_layer_norm = False
        self.persist_layer_norm = persist_layer_norm

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]

        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs)) if bias else None
        else:
            assert not bias, f"when elementwise_affine = False, bias item is None"
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.sequence_parallel_enabled = sequence_parallel_enabled
        if self.elementwise_affine:
            _set_sequence_parallel_enabled(self.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(self.bias, self.sequence_parallel_enabled)

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if fused_layer_norm is None or torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)

        if self.elementwise_affine:
            if self.persist_layer_norm:
                output = fast_layer_norm(input, self.weight, self.bias, self.eps)
                return make_viewless_tensor(
                    inp=output, requires_grad=input.requires_grad, keep_graph=True
                )
            else:
                return fused_layer_norm_affine(input, self.weight, self.bias, self.normalized_shape, self.eps)
        else:
            return fused_layer_norm(input, self.normalized_shape, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
