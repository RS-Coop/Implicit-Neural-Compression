'''
Model

LightningModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
'''

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics as tm

from .modules.metrics import R3Error, RPWError, RFError, PSNR
from .modules.siren import Siren
from .modules.wire import Wire
from .modules.loss import R3Loss, RPWLoss, W2Loss

# from .modules.injector import Injector

from .utils.sketch import sketch
from .utils.diff_ops import jacobian

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
            inr_type = "siren",
            hidden_features = 128,
            blocks = 1,
            loss_fn = "R3Error",
            learning_rate = 1e-4,
            scheduler = True,
            output_activation = "Identity",
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
        if loss_fn == "R3Loss":
            self.loss_fn = R3Loss()
        elif loss_fn == "RPWLoss":
            self.loss_fn = RPWLoss()
        elif loss_fn == "W2Loss":
            self.loss_fn = W2Loss()
        else:
            self.loss_fn = getattr(nn, loss_fn)()

        #Build INR network
        self.output_activation = getattr(nn, output_activation)()

        if inr_type == "siren":
            self.inr = Siren(input_shape[2], hidden_features, blocks, output_shape[2])
        elif inr_type == "wire":
            self.inr = Wire(input_shape[2], hidden_features, blocks, output_shape[2])
        else:
            raise Exception(f'Invalid inr_type {inr_type}')

        #Metrics
        self.error = R3Error(num_channels=output_shape[2])
            
        self.test_metrics = tm.MetricCollection([RPWError(num_channels=output_shape[2]), RFError(num_channels=output_shape[2]), PSNR(num_channels=output_shape[2])])

        self.prefix = ''
        self.denormalize = None

        #exact parameter count
        print(f"Exact parameter count: {self.size}")

        #continual backpro
        # self.injector = Injector(self.inr, replacement_rate=0.3)

        return
    
    @property
    def size(self):
        return sum(p.numel() for p in self.parameters())
    
    def unpack(self, batch):
        full = batch.get("full", (None, None))
        sketch = batch.get("sketch", (None, None, None))

        return full, sketch

    '''
    [Optional] A forward eavaluation of the network.
    '''
    def forward(self, coords):
        return self.output_activation(self.inr(coords))

        # c, output = self.inr(coords)

        # return c, self.output_activation(output)

    '''
    A single training step on the given batch.

    Output:
        torch loss
    '''
    def training_step(self, batch, idx):

        (c1, f1), (c2, f2, s) = self.unpack(batch)

        # if c2 is not None and idx%300==0:
        #     self.injector(c2) #update CBP parameters and weights

        l1 = self.loss_fn(self(c1), f1) if c1 is not None else torch.tensor([0.0], requires_grad=True, device=self.device)
        l2 = self.loss_fn(sketch(self(c2), s, device=self.device), f2) if c2 is not None else torch.tensor([0.0], requires_grad=True, device=self.device) #sketch loss

        loss = l1 + l2

        self.log('train_loss_1', l1, on_step=True, on_epoch=False, sync_dist=True, batch_size=f1.shape[0] if f1 is not None else 1)
        self.log('train_loss_2', l2, on_step=True, on_epoch=False, sync_dist=True, batch_size=f2.shape[0] if f2 is not None else 1)

        return loss

    '''
    [Optional] A single validation step.
    '''
    def validation_step(self, batch, idx):
        coords, features = batch

        #predictions
        preds = self(coords)
        # _, preds = self(coords)

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
        # _, preds = self(coords)

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
        self.log(self.prefix+'test_avg_max_err', torch.mean(max_err), on_step=False, on_epoch=True)

        #log other test metrics
        metric_dict = self.test_metrics.compute()
        
        for key, value in metric_dict.items():
            self.log(self.prefix+'test_avg_'+key, torch.mean(value), on_step=False, on_epoch=True)

        #per channel error
        if err.ndim != 0:
            for i in range(err.shape[0]):
                self.log(self.prefix+f"c_{i}_err", err[i], on_step=False, on_epoch=True)
                self.log(self.prefix+f"c_{i}_max", max_err[i], on_step=False, on_epoch=True)

                for key, value in metric_dict.items():
                    self.log(self.prefix+f'test_c_{i}_'+key, value[i], on_step=False, on_epoch=True)

        #reset metrics
        self.error.reset()
        self.test_metrics.reset()

        return

    '''
    [Optional] A single prediction step.
    '''
    def predict_step(self, batch, idx):
        coords, _ = batch

        # return self(coords)[1]
        return self(coords)

        # coords, _ = batch

        # c, preds = self(coords)

        # #compute jacobian
        # J = jacobian(preds, c)[:,:,:-1]

        # #concatenate with preds
        # preds = torch.cat((preds, torch.flatten(J, start_dim=1)), dim=1)

        # return preds

    '''
    Configure optimizers and optionally configure learning rate scheduler.

    Ouput:
        {torch optimizer, torch scheduler}
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.learning_rate)
    
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
