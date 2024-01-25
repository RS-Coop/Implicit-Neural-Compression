'''
Model

LightningModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
'''

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics as tm

from .metrics import r3error, R3Error, PSNR
from .modules import Siren

class R3Loss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

    def forward(self, input, target):
        return torch.mean(r3error(input, target))

'''
'''
class Model(LightningModule):
    '''
    Build model

    Input:
        loss_fn: model loss function for training
    '''
    def __init__(self,
            input_shape,
            output_shape,
            loss_fn = "MSELoss",
            learning_rate = 1e-2,
            scheduler = True,
            output_activation = "Tanh",
        ):
        super().__init__()

        #Log model hyperparameters
        self.save_hyperparameters(ignore=[])

        #Training hyperparameters
        self.learning_rate = learning_rate
        self.scheduler = scheduler

        #
        self.example_input_array = torch.zeros(input_shape)

        #Loss function
        # self.loss_fn = getattr(nn, loss_fn)()
        self.loss_fn = R3Loss()

        #Build SIREN
        self.output_activation = getattr(nn, output_activation)()

        self.siren = Siren(input_shape[1], 256, 3, output_shape[1], outermost_linear=True)

        #Metrics
        self.error = R3Error(num_channels=output_shape[1])
        self.test_metrics = tm.MetricCollection([PSNR()])

        self.prefix = ''
        self.denormalize = None

        return

    '''
    [Optional] A forward eavaluation of the network.
    '''
    def forward(self, coords):
        return self.output_activation(self.siren(coords))

    '''
    A single training step on the given batch.

    Output:
        torch loss
    '''
    def training_step(self, batch, idx):
        coords, features = batch

        preds = self(coords)

        loss = self.loss_fn(preds, features)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    '''
    [Optional] A single validation step.
    '''
    def validation_step(self, batch, idx):
        coords, features = batch

        #predictions
        preds = self(coords)

        #compute error
        self.error.update(preds, features)

        #log validation error
        self.log('val_err', self.error, on_step=False, on_epoch=True)

        return

    '''
    [Optional] A single test step.
    '''
    def test_step(self, batch, idx):
        coords, features = batch

        #predictions
        preds = self(coords)

        #update error
        if self.denormalize != None:
            preds = self.denormalize(preds)
            features = self.denormalize(features)

        self.error.update(preds, features)

        #log other metrics
        self.test_metrics.update(preds, features)

        return
    
    def on_test_epoch_end(self):
        #log average and max error w.r.t batch
        err = self.error.compute(reduce_channels=False)
        max_err = self.error.max

        self.log(self.prefix+'test_avg_err', torch.mean(err), on_step=False, on_epoch=True)
        self.log(self.prefix+'test_max_err', max_err, on_step=False, on_epoch=True)
        
        #per channel error
        if err.ndim != 0:
            for i, channel_error in enumerate(err):
                self.log(self.prefix+f"c_{i}_err", channel_error, on_step=False, on_epoch=True)

        #log other test metrics
        metric_dict = self.test_metrics.compute()
        
        for key, value in metric_dict.items():
            self.log(self.prefix+key, value, on_step=False, on_epoch=True)

        #reset metrics
        self.error.reset()
        self.test_metrics.reset()

        return

    '''
    [Optional] A single prediction step.
    '''
    def predict_step(self, batch, idx):
        coords, _ = batch

        return self(coords)

    '''
    Configure optimizers and optionally configure learning rate scheduler.

    Ouput:
        {torch optimizer, torch scheduler}
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.75, verbose=True)
            
            scheduler_config = {"scheduler": scheduler, "monitor": "val_err"}

            config = {"optimizer": optimizer, "lr_scheduler": scheduler_config}

        else:
            config = {"optimizer": optimizer}

        return config

    '''
    [Optional] Edit the checkpoint before loading.
    '''
    def on_load_checkpoint(self, checkpoint):

        # state_dict = checkpoint["state_dict"]

        return

    '''
    [Optional] Edit the checkpoint before saving.
    '''
    def on_save_checkpoint(self, checkpoint):

        # state_dict = checkpoint["state_dict"]

        return
