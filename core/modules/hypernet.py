'''
'''

import torch
import torch.nn as nn
import numpy as np

from contextlib import contextmanager

from .siren import Siren, SineBlock, SineLayer
from .wire import Wire

'''
'''
class Hypernet(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 blocks,
                 out_features
    ):
        super().__init__()

        #Network
        self.net = Siren(in_features, hidden_features, blocks, out_features)

    def initialize(self, target):
        with torch.no_grad():
            self.net.net[-1].weight *= 1e-2
            self.net.net[-1].bias.data = target

    def forward(self, input):
        input = input.view(1,1,-1)
        return self.net(input).view(-1)

'''
'''
class HyperINR(nn.Module):
    def __init__(self,
                hypernet_kwargs,
                inr_kwargs
        ):
        super().__init__()

        #inr
        self.inr = Siren(**inr_kwargs)
        p = nn.utils.parameters_to_vector(self.inr.parameters())

        #hypernet
        self.hypernet = Hypernet(**hypernet_kwargs, out_features=p.numel())
        self.hypernet.initialize(p)

    def format(self, params):
        state = self.inr.state_dict()
        c_idx = 0
        n_idx = 0
        for key, val in state.items():
            n_idx += val.numel()
            state[key] = params[c_idx:n_idx].view(val.shape)

            c_idx = n_idx

        return state

    def forward(self, t, xt):
        out = []
        
        for i, t_batch in enumerate(t):
            params = self.hypernet(t_batch)

            y = torch.func.functional_call(self.inr, self.format(params), xt[i,...])

            out.append(y)

        return torch.cat(out)