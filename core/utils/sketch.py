import torch
import numpy as np

def sketch(features, sketch_stuff, device='cpu'):
    seeds, rank = sketch_stuff

    seeds = seeds.reshape(-1)

    sf = []

    for i, seed in enumerate(seeds):
        sf.append(fjlt(features[i,...], seed, rank, device))

    return torch.stack(sf)

# def sketch(self, features, sketch_stuff):
#     seeds, self.rank = sketch_stuff
#     self.num_points = features.shape[1]

#     seeds = seeds.reshape(-1)

#     return torch.vmap(self.fjlt, randomness="different")(features, seeds)

#################################################
#Gaussian

def gaussian(f, seed, rank, device):
    torch.manual_seed(seed)
    sketch = torch.randn(f.shape[0], rank, device='cpu').to(device)
    sketch /= np.sqrt(rank)

    return torch.einsum('nc,nr->rc', f, sketch)

#################################################
#FJLT

# def sketch(features, sketch_stuff, device='cpu'):
#     seeds, rank = sketch_stuff

#     seeds = seeds.reshape(-1)

#     n = features.shape[1]

#     D = []
#     I = []

#     for i, seed in enumerate(seeds):
#         torch.manual_seed(seed)

#         D.append(torch.sign(torch.randn(n, device='cpu').to(device)).unsqueeze(1).expand(-1, features.shape[2]))
#         I.append((torch.randperm(n, device='cpu').to(device))[:rank])

#     D = torch.stack(D)
#     I = torch.stack(I)

#     return (np.sqrt(n/rank))*torch.vmap(_fjlt)(features, D, I)

# def _fjlt(f, d, ind):
#     return (dct((d*f).t()).t())[ind]

def fjlt(f, seed, rank, device):
    torch.manual_seed(seed)

    n = f.shape[0]

    d = torch.sign(torch.randn(n, device='cpu').to(device)).unsqueeze(1).expand_as(f)

    ind = (torch.randperm(n, device='cpu').to(device))[:rank]

    return (np.sqrt(n/rank)*dct((d*f).t()).t())[ind]

'''
https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
'''
def dct(x):
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    V[:, 0] /= np.sqrt(N) * 2
    V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V