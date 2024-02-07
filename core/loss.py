'''

'''

import torch
import torch.nn as nn
from .metrics import r3error
from .diff_ops import jacobian

'''
Root relative reconstruction loss
'''
class R3Loss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, preds, target):
        return torch.mean(r3error(preds, target))

'''
Order 2 Sobolev loss
'''
class W2Loss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, input, preds, target):

        #compute jacobian
        J = jacobian(preds, input)

        #concatenate with preds
        preds = torch.cat(preds, J.reshape(-1), dim=1)

        #compute error metric
        error = torch.functional.mse(preds, target)

        return error