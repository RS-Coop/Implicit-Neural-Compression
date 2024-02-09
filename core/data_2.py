'''
Data loading utilities.

LightningDataModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html?highlight=DataModule
'''

import pathlib
from warnings import warn

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule

'''
'''
class MeshDataset(Dataset):
    def __init__(self,
            points_path,
            features_path,
            channels,
            channels_last=True,
            normalize=True
        ):
        super().__init__()

        try:
            #Points
            self.p = torch.from_numpy(np.load(points_path).astype(np.float32))

            #Features
            f = torch.from_numpy(np.load(features_path).astype(np.float32))

            assert f.dim() == 3, f"Features has {f.dim()} dimensions, but should only have 3"

            #move to channels last
            if channels_last == False:
                f = torch.movedim(f, 1, 2)

            f = f[:,:,channels]

        except FileNotFoundError:
            raise Exception(f'Error loading points {points_path} and/or features {features_path}')

        except Exception as e:
            raise e
        
        if normalize:
            mean = torch.mean(f, dim=(0,1))
            stdv = torch.sqrt(torch.var(f, dim=(0,1)))

            f = (f-mean)/stdv

            max = torch.amax(torch.abs(f), dim=(0,1))

            f = f/max

            self.denormalize = lambda f_: stdv.to(f_.device)*f_*max.to(f_.device) + mean.to(f_.device)

            print("\nUsing clipped z-score normalization")

        self.f = f

        self.num_points = self.p.shape[0]
        self.num_snapshots = self.f.shape[0]

        return

    def __len__(self):
        return self.num_points*self.num_snapshots #time X num_points

    def __getitem__(self, idx):
        i = idx%self.num_points
        t = idx//self.num_points

        #normalized time
        t_coords = torch.tensor([t/self.num_snapshots, 0, self.num_snapshots-1])

        return torch.cat((self.p[i,:], t_coords, self.f[0,i,:], self.f[-1,i,:])), self.f[t,i,:]
    
    def getall(self, denormalize=True):

        if denormalize:
            features = self.denormalize(self.f)
        else:
            features = self.f 

        return features

'''
'''
class DataModule(LightningDataModule):
    '''
    Input:
        data_dir: path to dataset directory (usually absolute is more robust)
        batch_size: torch dataloader batch size
        num_workers: machine dependent, more workers means faster loading
    '''
    def __init__(self,
            spatial_dim,
            points_path,
            features_path,
            batch_size,
            channels,
            data_dir = "./",
            normalize = True,
            split = 0.8,
            shuffle = True,
            num_workers = 4,
            persistent_workers = True,
            pin_memory = True
        ):
        super().__init__()

        #channels
        if isinstance(channels, list):
            assert len(channels) != 0
        elif isinstance(channels, int):
            channels = [i for i in range(channels)]
        else:
            raise ValueError("Channels must be a list or an integer")
        
        args = locals()
        args.pop('self')

        for key, value in args.items():
            setattr(self, key, value)

        self.points_path = pathlib.Path(self.data_dir).joinpath(self.points_path)
        self.features_path = pathlib.Path(self.data_dir).joinpath(self.features_path)

        self.train, self.val, self.test, self.predict = None, None, None, None

        return
    
    @property
    def input_shape(self):
        return (1, self.spatial_dim+1+2+2*len(self.channels))

    @property
    def output_shape(self):
        return (1, len(self.channels))

    '''
    Load and preprocess data
    '''
    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize)

            train_size = round(self.split*len(train_val))
            val_size = len(train_val) - train_size

            self.train, self.val = random_split(train_val, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")
        
        return

    '''
    Used in Trainer.fit
    '''
    def train_dataloader(self):
        return DataLoader(self.train,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)

    '''
    Used in Trainer.fit
    '''
    def val_dataloader(self):
        return DataLoader(self.val,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Used in Trainer.test
    '''
    def test_dataloader(self):
        return DataLoader(self.test,
                            batch_size=self.test.num_points,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Used in Trainer.predict
    '''
    def predict_dataloader(self):
        return DataLoader(self.predict,
                            batch_size=self.predict.num_points,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
    '''
    [Optional] Clean up data
    '''
    def teardown(self, stage=None):
        pass
