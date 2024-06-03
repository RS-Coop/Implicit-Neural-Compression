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
                 out_features
    ):
        super().__init__()

        self.freq_base = 1.5
        self.freq_mult = 1.0
        self.freq_ceil = 5.0
        self.num_freqs = 1
        self.in_features = in_features

        #Network
        self.net = Siren(self.num_freqs, hidden_features, 2, out_features)

    def initialize(self, target):
        with torch.no_grad():
            self.net.net[-1].weight *= 1e-2
            self.net.net[-1].bias.data = target

    def positional_encoding(self, input):
            
        basic_lift = torch.tile(input, (self.num_freqs,))

        return basic_lift
    
    def random_fourier_features(self, input):

        sigma = 1 / np.sqrt(self.in_features)
        self.b = torch.randn(self.num_freqs*self.in_features, requires_grad=False, device=input.device) * sigma
        self.w = torch.rand(self.num_freqs*self.in_features, requires_grad=False, device=input.device) * 2 * torch.pi

        vp = (2 * np.pi * input * self.b.T) + self.w
        vp_out = torch.cos(vp)

        return vp_out

    def forward(self, input):
        input = input.view(1,1,1)
        return self.net(input).view(-1)

'''
'''
class HyperINR(nn.Module):
    def __init__(self,
                inr_type,
                input_size,
                inr_hidden_features,
                blocks,
                output_size,
                hyper_hidden_features
        ):
        super().__init__()

        self.inr = Siren(input_size, inr_hidden_features, blocks, output_size)
        self.hypernet = Hypernet(in_features=1, hidden_features=hyper_hidden_features, out_features=self.inr.size)

        self.hypernet.initialize(nn.utils.parameters_to_vector(self.inr.parameters()))

    def register(self, params):
        pass

    def update(self, params):
        c_idx = 0
        n_idx = 0
        for p in self.inr.parameters():
            n_idx += p.numel()
            p = params[c_idx:n_idx].view(p.shape)

            c_idx = n_idx

    def format(self, params):
        state = self.inr.state_dict()
        c_idx = 0
        n_idx = 0
        for key, val in state.items():
            n_idx += val.numel()
            state[key] = params[c_idx:n_idx].view(val.shape)

            c_idx = n_idx

        return state

    @contextmanager
    def using(self, params):
        # nn.utils.vector_to_parameters(params, self.inr.parameters()) #doesn't work because gradient doesn't flow
        # self.update(params) #doesn't work because gradient doesn't flow
        yield

    def forward(self, t, x):
        out = []
        for i, t_i in enumerate(t):
            params = self.hypernet(t_i)

            y = torch.func.functional_call(self.inr, self.format(params), x[[i],...])

            # with self.using(params):
            #     out.append(self.inr(x[[i],...]))

            out.append(y)

        return torch.cat(out)