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
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.initialize()

    @property
    def weight(self):
        return self.linear.weight
    
    @property
    def bias(self):
        return self.linear.bias
    
    def init_weights(self, m):
        if self.is_first:
            r = 1/self.in_features
        else:
            r = np.sqrt(6/self.in_features)/self.omega_0

        m.uniform_(-r, r)
        
        return
    
    def initialize(self):
        with torch.no_grad():
            self.init_weights(self.linear.weight)
        
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

        # mid_width = int(width/2)

        # self.net = nn.Sequential(SineLayer(width, mid_width, bias=bias, omega_0=omega_0),
        #                          SineLayer(mid_width, mid_width, bias=bias, omega_0=omega_0),
        #                          SineLayer(mid_width, width, bias=bias, omega_0=omega_0))
        
        self.net = nn.Sequential(SineLayer(width, width, bias=bias, omega_0=omega_0),
                                 SineLayer(width, width, bias=bias, omega_0=omega_0))

    def forward(self, input):
        output = self.net(input)

        return 0.5*(input+output)
        
'''
''' 
class Siren(nn.Module):
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
        self.net.append(nn.Flatten(start_dim=0, end_dim=-2))

        #first layer
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0))

        #blocks
        for i in range(blocks):
            self.net.append(SineBlock(hidden_features, omega_0=hidden_omega_0))

        #last layer
        if outermost_linear:
            last = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                r = np.sqrt(6/hidden_features)/hidden_omega_0
                last.weight.uniform_(-r, r)
                
        else:
            last = SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0)

        self.net.append(last)

    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_hidden_layers(self):
        return (len(self.net)-3)

    def get_layers(self):
        for layer in self.net[2:]:
            if isinstance(layer, SineBlock):
                for l in layer.net:
                    yield l
            else:
                yield layer
            
        return
    
    def activations(self, coords):
        activations = []
        out = self.net[:2](coords)
        activations.append(out)
        for i in range(2, len(self.net)-1):
            out = self.net[i](out)
            activations.append(out)
        
        return activations
    
    def forward(self, coords):

        shape = coords.shape
        out = self.net(coords)

        return torch.unflatten(out, 0, shape[0:-1])
    
        # coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        # output = self.net(coords)
        # return coords, output

class INR(nn.Module):
    def __init__(self,
                 inr_kwargs
        ):
        super().__init__()

        self.siren = Siren(**inr_kwargs)

    @property
    def size(self):
        return self.siren.size
    
    def forward(self, t, x):
        t = torch.flatten(t, start_dim=0, end_dim=1)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        coords = torch.cat((x, t.view(*t.shape,1,1).expand(-1, x.shape[1], -1)), dim=2)

        return self.siren(coords)

