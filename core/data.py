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
class DataModule(LightningDataModule):
    '''
    Input:
        data_dir: path to dataset directory (usually absolute is more robust)
        batch_size: torch dataloader batch size
        num_workers: machine dependent, more workers means faster loading
    '''
    def __init__(self
            data_dir,
            batch_size,
            shuffle = False,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True
        ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

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
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers*self.trainer.num_devices
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)

    '''
    Used in Trainer.fit
    '''
    def val_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Used in Trainer.test
    '''
    def test_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Used in Trainer.predict
    '''
    def predict_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Clean up data
    '''
    def teardown(self, stage=None):
        pass
