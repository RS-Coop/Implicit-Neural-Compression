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
from .modules.buffer import Buffer

'''
'''
class MeshDataset(Dataset):
    def __init__(self,
            points_path,
            features_path,
            channels,
            channels_last=True,
            normalize=True,
            gradients=False,
            index_time=True
        ):
        super().__init__()

        try:
            #Points
            self.points = torch.from_numpy(np.load(points_path).astype(np.float32))

            #Features
            features = torch.from_numpy(np.load(features_path).astype(np.float32))

            assert features.dim() == 3, f"Features has {features.dim()} dimensions, but should only have 3"

            #move to channels last
            if channels_last == False:
                features = torch.movedim(features, 1, 2)

            features = features[:,:,channels]

            if gradients:
                g_path = features_path.with_stem(features_path.stem+"_gradients")
                gradients = torch.from_numpy(np.load(g_path).astype(np.float32))

                assert gradients.dim() == 4 and gradients.shape[-1] == self.points.shape[1], f"Gradients have incorrect shape"

                self.gradients = torch.flatten(gradients[:,:,channels,:], start_dim=2)
            else:
                self.gradients = None

        except FileNotFoundError:
            raise Exception(f'Error loading points {points_path} and/or features {features_path}')

        except Exception as e:
            raise e
        
        #normalize points
        mx, mi = torch.aminmax(self.points, dim=0)
        self.points = 2*(self.points-mi)/(mx-mi)-1

        self.denorm_p = lambda p: ((p+1)/2)*(mx-mi).to(p.device) + mi.to(p.device)
        
        #normalize features
        if normalize != False:

            if normalize == "z-score":
                mean = torch.mean(features, dim=(0,1))
                stdv = torch.sqrt(torch.var(features, dim=(0,1)))

                features = (features-mean)/stdv

                max = torch.amax(torch.abs(features), dim=(0,1))

                features = features/max

                self.denorm_f = lambda f: stdv.to(f.device)*f*max.to(f.device) + mean.to(f.device)

                print("\nUsing clipped z-score normalization")

            elif normalize == "0-1":
                mi_f, mx_f = torch.amin(features, dim=(0,1)), torch.amax(features, dim=(0,1))
                features = (features-mi_f)/(mx_f-mi_f)

                self.denorm_f = lambda f: (mx_f-mi_f).to(f.device)*f + mi_f.to(f.device)

                print("\nUsing 0-1 normalization")

            elif normalize == "-1-1":
                mi_f, mx_f = torch.amin(features, dim=(0,1)), torch.amax(features, dim=(0,1))
                features = 2*(features-mi_f)/(mx_f-mi_f)-1

                self.denorm_f = lambda f: (mx_f-mi_f).to(f.device)*((f+1)/2) + mi_f.to(f.device)

                print("\nUsing -1-1 normalization")

            else:
                raise Exception(f"Normalization type {normalize} is not valid.")
            
        else:
            self.denorm_f = lambda f: f

            print("\nNot normalizing features")

        self.features = features

        self.num_points = self.points.shape[0]
        self.num_snapshots = self.features.shape[0]

        if index_time:
            self.num_samples = self.num_snapshots
        else:
            self.num_samples = self.num_snapshots*self.num_points

        self.index_time = index_time

        return

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.index_time:
            #normalized time
            t_coord = torch.tensor(2*(idx/(self.num_snapshots-1))-1).expand(self.num_points, 1)

            #coordinates
            coordinates = torch.cat((self.points,t_coord))

            #features
            if self.gradients != None:
                features = torch.cat((self.features[idx,:,:], self.gradients[idx,:,:]))
            else:
                features = self.features[idx,:,:]

            return coordinates, features
        
        else:
            i = idx%self.num_points
            t = idx//self.num_points

            #normalized time
            t_coord = torch.tensor(2*(t/(self.num_snapshots-1))-1).unsqueeze(0)

            return torch.cat((self.points[i,:],t_coord)), self.features[t,i,:]
    
    
    def __getitems__(self, idxs):
        if self.index_time:
            #normalized time
            t_coord = (2*(torch.tensor(idxs)/(self.num_snapshots-1))-1).view(-1,1,1).expand(-1, self.num_points, -1)

            #coordinates
            coordinates = torch.cat((self.points.expand(len(idxs), -1, -1), t_coord), dim=2)

            #features
            if self.gradients != None:
                features = torch.cat((self.features[idxs,:,:], self.gradients[idxs,:,:]), dim=2)
            else:
                features = self.features[idxs,:,:]

            return torch.flatten(coordinates, end_dim=1), torch.flatten(features, end_dim=1)
        
        else:
            idxs = torch.tensor(idxs)

            i = idxs%self.num_points
            t = idxs//self.num_points

            #normalized time
            t_coord = (2*(t/(self.num_snapshots-1))-1).unsqueeze(1)

            return torch.cat((self.points[i,:],t_coord), dim=1), self.features.view(len(self),-1)[idxs,:]
    
    def get_points(self, denormalize=True):

        if denormalize:
            points = self.denorm_p(self.points)
        else:
            points = self.points

        return points
    
    def get_features(self, denormalize=True):

        if denormalize:
            features = self.denorm_f(self.features)
        else:
            features = self.features 

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
            buffer = {},
            gradients = False,
            data_dir = "./",
            normalize = True,
            index_time = False,
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
        return (1, self.spatial_dim+1)

    @property
    def output_shape(self):
        return (1, len(self.channels))

    '''
    Load and preprocess data
    '''
    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=self.gradients, index_time=self.index_time)

            train_size = round(self.split*len(train_val))
            val_size = len(train_val) - train_size

            self.train, self.val = random_split(train_val, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=False, index_time=True)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=False, index_time=True)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")
        
        return

    '''
    Used in Trainer.fit
    '''
    def train_dataloader(self):
        if len(self.buffer) != 0:
            batch_sampler = Buffer(self.train.dataset.num_snapshots, **self.buffer, batch_size=self.batch_size)

            return DataLoader(self.train,
                            batch_sampler=batch_sampler,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory)
        else:
            return DataLoader(self.train,
                            batch_size=self.batch_size,
                            shuffle=self.shuffle,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=lambda x: x)

    '''
    Used in Trainer.fit
    '''
    def val_dataloader(self):
        return DataLoader(self.val,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=lambda x: x)
    '''
    [Optional] Used in Trainer.test
    '''
    def test_dataloader(self):
        return DataLoader(self.test,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=lambda x: x)
    '''
    [Optional] Used in Trainer.predict
    '''
    def predict_dataloader(self):
        return DataLoader(self.predict,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers*self.trainer.num_devices,
                            persistent_workers=self.persistent_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=lambda x: x)
    '''
    [Optional] Clean up data
    '''
    def teardown(self, stage=None):
        pass
