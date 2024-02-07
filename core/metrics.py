'''
Custom metrics subclassed from TorchMetrics

TorchMetrics documentation:
    https://torchmetrics.readthedocs.io/en/latest/

TorchMetrics example:
    https://torchmetrics.readthedocs.io/en/stable/pages/implement.html
'''

import torch
from torchmetrics import Metric

################################################################################

'''
Root Relative Reconstruction (R3) Error.

Input:
    preds, target: tensors of shape (B, C)
'''
def r3error(preds, targets):
    assert preds.shape == targets.shape, f"{preds.shape} does not equal {targets.shape}"

    n = torch.sum((preds-targets)**2, dim=(0))
    d = torch.sum((targets)**2, dim=(0))

    return torch.sqrt(n/d)

class R3Error(Metric):
    
    #metric attributes
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_channels):
        super().__init__()
        self.add_state("error", default=torch.zeros(num_channels), dist_reduce_fx="sum")
        self.add_state("max", default=torch.tensor(0.0), dist_reduce_fx="max")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        err = r3error(preds, targets)

        self.error += err
        self.max = torch.maximum(torch.mean(err), self.max)
        self.num_samples += 1

    def compute(self, reduce_channels=True):
        err = self.error/self.num_samples

        if reduce_channels: err = torch.mean(err)

        return err

################################################################################

# '''
# '''
# def w2error(preds, targets):
#     pass

# class W2Error(Metric):

#     #metric attributes
#     is_differentiable = False
#     higher_is_better = False
#     full_state_update = False

#     def __init__(self, num_channels):
#         super().__init__()
#         self.add_state("error", default=torch.zeros(num_channels), dist_reduce_fx="sum")
#         self.add_state("max", default=torch.tensor(0.0), dist_reduce_fx="max")
#         self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

################################################################################

'''
Peak Signal to Noise Ratio

Input:
    preds, target: tensors of shape (B, C)
'''
def psnr(preds, targets):
    r = torch.amax(targets, dim=(0))
    mse = torch.mean((preds-targets)**2, dim=(0))

    return torch.mean(10*torch.log10(r**2/mse))

class PSNR(Metric):

    #metric attributes
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        self.psnr += psnr(preds, targets)
        self.num_samples += 1

    def compute(self):
        return self.psnr/self.num_samples