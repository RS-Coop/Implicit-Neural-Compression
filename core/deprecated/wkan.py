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
class WKLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features
        ):
        super().__init__()
        
        self.in_features = in_features

        self.shift = nn.Parameter(torch.zeros(in_features, out_features), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(in_features, out_features)*3.0, requires_grad=True)
        self.freq = nn.Parameter(torch.ones(in_features, out_features)*10.0, requires_grad=True)
        
    def forward(self, input):
        x_shift = input.unsqueeze(2) - self.shift.expand(input.shape[0], -1, -1)

        return torch.sum(torch.exp(-(x_shift**2)*self.scale**2)*torch.cos(self.freq*x_shift), dim=1)
    
'''
''' 
class WKAN(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 blocks,
                 out_features
        ):
        super().__init__()

        #network
        self.net = nn.Sequential()

        #flatten
        self.net.append(nn.Flatten(start_dim=0, end_dim=1))

        #first layer
        self.net.append(WKLayer(in_features, hidden_features))

        #blocks
        for i in range(blocks):
            self.net.append(WKLayer(hidden_features, hidden_features))

        #last layer
        self.net.append(WKLayer(hidden_features, out_features))
    
    def forward(self, coords):

        shape = coords.shape
        out = self.net(coords)

        return torch.unflatten(out, 0, shape[0:2])