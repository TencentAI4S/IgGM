# Copyright (c) 2024, Tencent Inc. All rights reserved.
# Data: 2024/5/29 17:47

import torch.nn as nn

from IgGM.model.layer import LayerNorm, Linear


class OuterProductMeanSS(nn.Module):
    """Outer-produce mean for transforming single features into an update for pair features."""

    def __init__(self, c_s, c_z, c_hidden=32):
        super(OuterProductMeanSS, self).__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.layer_norm = LayerNorm(c_s)
        self.linear_1 = Linear(c_s, c_hidden)
        self.linear_2 = Linear(c_s, c_hidden)
        self.linear_out = Linear(c_hidden ** 2, c_z, init='final')

    def forward(self, s):
        """
        Args:
            s: [N, L, c_s], single features

        Returns:
            z: update term for pair features of size N x L x L x c_z
        """

        N, L, _ = s.shape  # pylint: disable=invalid-name
        s = self.layer_norm(s)
        afea_tns = self.linear_1(s).view(N, L, 1, self.c_hidden, 1)
        bfea_tns = self.linear_2(s).view(N, 1, L, 1, self.c_hidden)
        ofea_tns = (afea_tns * bfea_tns).view(N, L, L, self.c_hidden ** 2)

        return self.linear_out(ofea_tns)
