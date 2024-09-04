'''

'''

import torch
import torch.nn as nn
from .metrics import r3error, rpwerror
from ..utils.diff_ops import jacobian

'''
Root relative reconstruction loss
'''
class R3Loss(nn.Module):

    def __init__(self, reduction="mean", dim=1):
        super().__init__()

        self.reduction = reduction
        self.dim = dim

    def forward(self, preds, target):
        return torch.mean(r3error(preds, target, dim=self.dim))
    
'''
Point-wise relative reconstruction loss
'''
class RPWLoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, preds, target):
        return torch.mean(rpwerror(preds, target))

'''
Order 2 Sobolev loss
'''
class W2Loss(nn.Module):

    def __init__(self, reduction="mean", time_diff=False):
        super().__init__()

        self.reduction = reduction
        self.time_diff = time_diff

    def forward(self, input, preds, target):

        #compute jacobian
        J = jacobian(preds, input)

        #don't include time derivative
        if not self.time_diff:
            J = J[:,:,:-1]

        #concatenate with preds
        preds = torch.cat((preds, torch.flatten(J, start_dim=1)), dim=1)

        #compute error metric
        error = torch.nn.functional.mse_loss(preds, target)

        return error