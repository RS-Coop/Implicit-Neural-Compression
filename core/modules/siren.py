'''
Independent LightningModule constituents.

e.g. encoder and decoder for an autoencoder
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

'''
https://github.com/vsitzmann/siren/blob/master/modules.py
'''

'''
'''
class SineLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30
        ):
        super().__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                r = 1/self.in_features
            else:
                r = np.sqrt(6/self.in_features)/self.omega_0

            self.linear.weight.uniform_(-r, r)
        
    def forward(self, input):
        return torch.sin(self.omega_0*self.linear(input))

'''
''' 
class SineBlock(nn.Module):
    def __init__(self,
                 width,
                 bias=True,
                 omega_0=30
        ):
        super().__init__()

    #     self.net = nn.Sequential(SineLayer(width, width, bias=bias, omega_0=omega_0),
    #                              SineLayer(width, width, bias=bias, omega_0=omega_0))

        self.net = SineLayer(width, width, bias=bias, omega_0=omega_0)

    # def forward(self, input):
    #     output = self.net(input)

    #     return 0.5*(input+output)

    def forward(self, input):
        return self.net(input)
        
'''
''' 
class Siren(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 blocks,
                 out_features,
                 outermost_linear=False, 
                 first_omega_0=30, 
                 hidden_omega_0=30
        ):
        super().__init__()

        #first layer
        self.first = SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)

        #blocks
        self.blocks = nn.Sequential()
        for i in range(blocks):
            self.blocks.append(SineBlock(hidden_features, omega_0=hidden_omega_0))

        #last layer
        if outermost_linear:
            last = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                r = np.sqrt(6/hidden_features)/hidden_omega_0
                last.weight.uniform_(-r, r)
                
        else:
            last = SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)

        self.last = last

    def net(self, input):
        return self.last(self.blocks(self.first(input)))
    
    def forward(self, coords):
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # output = self.net(coords)
        # return coords, output

        return self.net(coords)

####################################################################

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(30 * input)

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)