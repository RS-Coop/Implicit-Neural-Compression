'''
Data loading utilities.

LightningDataModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html?highlight=DataModule
'''

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

'''
'''
class DataModule1(LightningDataModule):
    '''
    Input:
        data_dir: path to dataset directory (usually absolute is more robust)
        batch_size: torch dataloader batch size
        num_workers: machine dependent, more workers means faster loading
    '''
    def __init__(self
                    data_dir,
                    batch_size,
                    num_workers=1):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    '''
    Load and preprocess data
    '''
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train, self.val = None, None

        elif stage == "test" or stage is None:
            self.test = None

        elif stage == "predict" or stage is None:
            self.predict = None

        else:
            raise ValueError("Invalid stage.")

    '''
    Used in Trainer.fit
    '''
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    '''
    Used in Trainer.fit
    '''
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    '''
    [Optional] Used in Trainer.test
    '''
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    '''
    [Optional] Used in Trainer.predict
    '''
    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers)

    '''
    [Optional] Clean up data
    '''
    def teardown(self, stage=None):
        pass
