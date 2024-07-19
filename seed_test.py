import torch

seeds = [1, 2, 3]

for seed in seeds:
    torch.manual_seed(seed)

    t = torch.randn(10, device='cpu')

    print(t)