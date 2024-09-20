# -*- coding: utf-8 -*-
# Copyright (c) 2024, Tencent Inc. All rights reserved.
from typing import Callable, Union

import torch.nn as nn

from ..utils.init_weights import INIT_METHOD_REGISTRY


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            init: Union[str, Callable, None] = None,
            device=None,
            dtype=None
    ):
        """
        Args:
            in_features:
                The final dimension of inputs to the layer
            out_features:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:
                None: default init
                "lecun": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.init = init
        super(Linear, self).__init__(in_features, out_features,
                                     bias=bias, **factory_kwargs)

    def reset_parameters(self) -> None:
        init = self.init
        assert isinstance(init, (str, Callable, type(None)))
        if isinstance(init, str):
            INIT_METHOD_REGISTRY[init](self.weight, self.bias)
        elif isinstance(init, Callable):
            return init(self.weight, self.bias)
        else:
            super().reset_parameters()
