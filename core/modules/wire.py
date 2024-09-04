'''
Independent LightningModule constituents.

e.g. encoder and decoder for an autoencoder
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

'''
'''
class WaveletLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=10.0,
                 sigma_0=3.0
        ):
        super().__init__()

        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.is_first = is_first
        
        self.in_features = in_features

        self.freqs = nn.Linear(in_features, out_features, bias=bias)
        self.scale = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.sigma_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))

'''
''' 
class WaveletBlock(nn.Module):
    def __init__(self,
                 width,
                 bias=True
        ):
        super().__init__()

        self.net = nn.Sequential(WaveletLayer(width, width, bias=bias),
                                 WaveletLayer(width, width, bias=bias))

    def forward(self, input):
        output = self.net(input)

        return 0.5*(input+output)
        
'''
''' 
class Wire(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 blocks,
                 out_features,
                 outermost_linear=True, 
                 first_omega_0=30, 
                 hidden_omega_0=30
        ):
        super().__init__()

        #network
        self.net = nn.Sequential()

        #flatten
        self.net.append(nn.Flatten(start_dim=0, end_dim=1))

        #first layer
        self.net.append(WaveletLayer(in_features, hidden_features, is_first=True))

        #blocks
        for i in range(blocks):
            self.net.append(WaveletBlock(hidden_features))

        #last layer
        if outermost_linear:
            last = nn.Linear(hidden_features, out_features)
        else:
            last = WaveletLayer(hidden_features, out_features, is_first=False)

        self.net.append(last)

    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, coords):
        shape = coords.shape
        out = self.net(coords)

        return torch.unflatten(out, 0, shape[0:2])