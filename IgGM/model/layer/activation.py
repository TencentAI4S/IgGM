import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_fn(activation: str):
    """ Returns the activation function corresponding to `activation` """
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return GELU()
    elif activation == 'relu_squared':
        return ReluSquared()
    elif activation == 'silu':
        return nn.SiLU()
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'linear':
        return lambda x: x
    else:
        raise RuntimeError('--activation-fn {} not supported'.format(activation))


class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class GELU(nn.Module):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        assert approximate in ['none', 'tanh', 'sigmoid', 'erf']
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'sigmoid':
            return torch.sigmoid(1.702 * x) * x
        elif self.approximate == 'erf':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        else:
            return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(self.approximate)
