'''
Model

LightningModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
'''

import torch
from pytorch_lightning import LightningModule

from .modules import Module1

'''
'''
class Model(LightningModule):
    '''
    Build model

    Input:
        loss_fn: model loss function for training
    '''
    def __init__(self,
                    loss_fn):
        super().__init__()

        #Log model hyperparameters
        self.save_hyperparameters(ignore=['loss_fn'])

    '''
    [Optional] A forward eavaluation of the network.
    '''
    def forward(self, x):
        pass

    '''
    A single training step on the given batch.

    Output:
        Torch loss
    '''
    def training_step(self, batch, idx):
        pass

    '''
    [Optional] A single validation step.
    '''
    def validation_step(self, batch, idx):
        pass

    '''
    [Optional] A single test step.
    '''
    def test_step(self, batch, idx):
        pass

    '''
    [Optional] A single prediction step.
    '''
    def predict_step(self, batch, idx):
        pass

    '''
    Configure optimizers and optionally configure learning rate scheduler.

    Ouput:
        Torch optimizer
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters)

        return optimizer
