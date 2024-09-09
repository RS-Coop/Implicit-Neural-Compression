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
    preds, target: tensors of shape (T, N, C)
'''
def r3error(preds, targets, dim=1):
    assert preds.shape == targets.shape, f"{preds.shape} does not equal {targets.shape}"

    n = torch.sum((preds-targets)**2, dim=dim)
    d = torch.sum((targets)**2, dim=dim)

    return torch.sqrt(n/(d+1e-3))

class R3Error(Metric):
    
    #metric attributes
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_channels):
        super().__init__()
        self.add_state("error", default=torch.zeros(num_channels), dist_reduce_fx="sum")
        self.add_state("max", default=torch.zeros(num_channels), dist_reduce_fx="max")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        err = r3error(preds, targets)

        self.error += torch.sum(err, dim=0)
        self.max = torch.maximum(torch.amax(err, dim=0), self.max)
        self.num_samples += err.shape[0]

    def compute(self, reduce_channels=True):
        err = self.error/self.num_samples

        if reduce_channels: err = torch.mean(err)

        return err
    
################################################################################

'''
Relative point-wise reconstruction error.

Input:
    preds, target: tensors of shape (T, N, C)
'''
def rpwerror(preds, targets):
    assert preds.shape == targets.shape, f"{preds.shape} does not equal {targets.shape}"

    error = (preds-targets)**2/(targets**2+1e-8)

    return error

class RPWError(Metric):
    
    #metric attributes
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_channels):
        super().__init__()
        self.add_state("error", default=torch.zeros(num_channels), dist_reduce_fx="sum")
        self.add_state("max", default=torch.zeros(num_channels), dist_reduce_fx="max")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        err = rpwerror(preds, targets)

        self.error += torch.sum(err, dim=(0,1))
        self.max = torch.maximum(torch.amax(err, dim=(0,1)), self.max)
        self.num_samples += err.shape[0]*err.shape[1]

    def compute(self, reduce_channels=False):
        err = self.error/self.num_samples

        if reduce_channels: err = torch.mean(err)

        return err
    
################################################################################

'''
Relative Frobenius reconstruction error.
'''
class RFError(Metric):
    
    #metric attributes
    is_differentiable = True
    higher_is_better = False
    full_state_update = False

    def __init__(self, num_channels):
        super().__init__()
        self.add_state("N", default=torch.zeros(num_channels), dist_reduce_fx="sum")
        self.add_state("D", default=torch.zeros(num_channels), dist_reduce_fx="sum")

    '''
    Input:
        preds, targets: tensors of shape (T, N, C)
    '''
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        self.N += torch.sum((preds-targets)**2, dim=(0,1))
        self.D += torch.sum((targets)**2, dim=(0,1))

    def compute(self, reduce_channels=False):
        err = torch.sqrt(self.N/self.D)

        if reduce_channels: err = torch.mean(err)

        return err

################################################################################

'''
Peak Signal to Noise Ratio

Input:
    preds, target: tensors of shape (T, N, C)
'''
def psnr(preds, targets, dim=1):
    r = torch.amax(targets, dim=dim)
    mse = torch.mean((preds-targets)**2, dim=dim)

    return 10*torch.log10((r**2+1e-8)/mse)

class PSNR(Metric):

    #metric attributes
    is_differentiable = True
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_channels):
        super().__init__()
        self.add_state("sum", default=torch.zeros(num_channels), dist_reduce_fx="sum")
        self.add_state("num_samples", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        
        val = psnr(preds, targets)

        self.sum += torch.sum(val, dim=0)
        self.num_samples += val.shape[0]

    def compute(self, reduce_channels=False):
        psnr_mean = self.sum/self.num_samples

        if reduce_channels: psnr_mean = torch.mean(psnr_mean)

        return psnr_mean