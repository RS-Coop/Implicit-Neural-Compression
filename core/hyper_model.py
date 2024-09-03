'''
Model

LightningModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule
'''

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics as tm

import copy

from core.modules.metrics import R3Error, RPWError, RFError, PSNR
from core.modules.loss import R3Loss, RPWLoss, W2Loss

from core.modules.hypernet import HyperINR

from core.utils.sketch import sketch
from core.utils.diff_ops import jacobian

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
            hypernet_kwargs,
            inr_kwargs,
            loss_fn = "R3Error",
            learning_rate = 1e-4,
            scheduler = False,
            output_activation = "Identity",
            sketch_type = "fjlt"
        ):
        super().__init__()

        #Log model hyperparameters
        self.save_hyperparameters(ignore=[])

        #Training hyperparameters
        self.learning_rate = learning_rate
        self.scheduler = scheduler

        #Unpack input shape
        hypernet_input_shape, inr_input_shape = input_shape

        #Sample input
        self.example_input_array = {'coords': (torch.zeros(hypernet_input_shape), torch.zeros(inr_input_shape))}

        #Loss function
        if loss_fn == "R3Loss":
            self.loss_fn = R3Loss()
        elif loss_fn == "RPWLoss":
            self.loss_fn = RPWLoss()
        elif loss_fn == "W2Loss":
            self.loss_fn = W2Loss()
        else:
            self.loss_fn = getattr(nn, loss_fn)()

        #Build network
        self.output_activation = getattr(nn, output_activation)()

        hypernet_kwargs["in_features"] = hypernet_input_shape[1]

        inr_kwargs["in_features"] = inr_input_shape[3]
        inr_kwargs["out_features"] = output_shape[2]

        self.hyper_inr = HyperINR(hypernet_kwargs, inr_kwargs)

        #Metrics
        self.error = R3Error(num_channels=output_shape[2])
            
        self.test_metrics = tm.MetricCollection([RPWError(num_channels=output_shape[2]), RFError(num_channels=output_shape[2]), PSNR(num_channels=output_shape[2])])

        self.prefix = ''
        self.denormalize = None

        #Sketch type
        self.sketch_type = sketch_type

        #Hypernetwork checkpoint
        self.hypernet_checkpoint = None

        #exact parameter count
        print(f"Exact parameter count: {self.size}")

        return
    
    @property
    def size(self):
        return sum(p.numel() for p in self.hyper_inr.hypernet.parameters())
    
    def unpack(self, batch):
        full = batch.get("full", (None, None))
        sketch = batch.get("sketch", (None, None, None))

        return full, sketch

    '''
    [Optional] A forward eavaluation of the network.
    '''
    def forward(self, coords):
        t, xt = coords

        return self.output_activation(self.hyper_inr(t, xt))

    '''
    A single training step on the given batch.

    Output:
        torch loss
    '''
    def training_step(self, batch, idx):

        # if idx%200 == 0:
        #     print("CHECKPOINTING")
        #     self.checkpoint()

        (c1, f1), (c2, f2, s) = self.unpack(batch)

        l1 = self.loss_fn(self(c1), f1) if c1 is not None else torch.tensor([0.0], requires_grad=True, device=self.device)
        l2 = self.loss_fn(sketch(self(c2), s, sketch_type=self.sketch_type, device=self.device), f2) if c2 is not None else torch.tensor([0.0], requires_grad=True, device=self.device) #sketch loss
        # l3 = self.compute_reg(c2[0]) if c2 is not None else torch.tensor([0.0], requires_grad=True, device=l1.device) #hypernet output loss

        loss = l1 + l2
        # loss = l1 + 10*l3
        # loss = l1+l2+l3

        self.log('train_loss_1', l1, on_step=True, on_epoch=False, sync_dist=True, batch_size=f1.shape[0] if f1 is not None else 1)
        self.log('train_loss_2', l2, on_step=True, on_epoch=False, sync_dist=True, batch_size=f2.shape[0] if f2 is not None else 1)
        # self.log('train_loss_3', l3, on_step=True, on_epoch=False, sync_dist=True, batch_size=f2.shape[0] if f2 is not None else 1)

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

        return self(coords)

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

        state_dict = checkpoint["state_dict"]

        if self.hypernet_checkpoint is not None:
            for key in list(state_dict.keys()):
                if "hypernet_checkpoint" in key:
                    state_dict.pop(key)

        return
    
    '''
    Hypernetwork direct regularization
    '''
    def checkpoint(self):
        
        self.hypernet_checkpoint = copy.deepcopy(self.hyper_inr.hypernet)

        return
    
    def compute_reg(self, t):

        losses = [torch.zeros(1, requires_grad=True, device=t.device) for i in range(t.shape[0])]

        for i, t_batch in enumerate(t):
            with torch.no_grad():
                ref = self.hypernet_checkpoint(t_batch)

            losses[i] = torch.nn.functional.l1_loss(ref, self.hyper_inr.hypernet(t_batch))

        return sum(losses)
