'''
Differential operators in PyTorch

https://github.com/vsitzmann/siren/blob/master/diff_operators.py
'''

import torch
import torch.autograd as AD

'''
Gradient of outputs (y) with respect to inputs (x).
'''
def jacobian(y, x):
    
    batch_size = y.shape[0]
    in_dim = x.shape[-1]
    out_dim = y.shape[-1]

    J = torch.zeros(batch_size, out_dim, in_dim, device=y.device)

    for i in range(out_dim):
        y_flat = y[...,i].view(-1, 1)
        J[:,i,:] = AD.grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    return J

'''

'''
def divergence(y, x, ):
    pass

'''

'''
def laplacian(y, x, ):
    pass
