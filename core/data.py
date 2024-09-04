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
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from .modules.sampler import Buffer

from .utils.sketch import sketch

'''
'''
class MeshDataset(Dataset):
    def __init__(self,
            points_path,
            features_path,
            channels,
            channels_last = True,
            normalize = True,
            gradients = False
        ):
        super().__init__()

        try:
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

                assert gradients.dim() == 4, f"Gradients have incorrect shape"

                # self.gradients = torch.flatten(gradients[:,:,channels,:], start_dim=2)
                self.gradients = gradients
            else:
                self.gradients = None

            #Points
            points = torch.from_numpy(np.load(points_path).astype(np.float32))

            if points.dim() == 2:
                self.points = points.expand(features.shape[0], -1, -1)
            elif points.dim() == 3:
                self.points = points
            else:
                raise Exception(f'Points has incorrect number of dimensions')

        except FileNotFoundError:
            raise Exception(f'Error loading points {points_path} and/or features {features_path}')

        except Exception as e:
            raise e
        
        #normalize points
        mx = torch.amax(self.points, dim=(0,1))
        mi = torch.amin(self.points, dim=(0,1))
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

        self.num_points = self.points.shape[1]
        self.num_snapshots = self.features.shape[0]

        return
    
    @property
    def size(self):
        return self.features.numel()

    def __len__(self):
        return self.num_snapshots
    
    def __getitems__(self, idxs):
        if isinstance(idxs, list): idxs = torch.tensor(idxs)

        if idxs.numel() == 0: return None, None

        #normalized time
        t_coord = (2*idxs/(self.num_snapshots-1)-1).view(-1,1,1).expand(-1, self.num_points, -1)

        #coordinates
        coordinates = torch.cat((self.points[idxs,:,:], t_coord), dim=2)

        #features
        if self.gradients != None:
            # features = torch.cat((self.features[idxs,:,:], torch.flatten(self.gradients[idxs,...], start_dim=2)), dim=2)
            features = self.features[idxs,:,:]
        else:
            features = self.features[idxs,:,:]

        return coordinates, features
    
    def get_points(self, denormalize=True):

        if denormalize:
            points = self.denorm_p(self.points)
        else:
            points = self.points

        return points[0,:,:] #NOTE: Assuming points are static across time
    
    def get_features(self, denormalize=True):

        if denormalize:
            features = self.denorm_f(self.features)
        else:
            features = self.features 

        return features
    
#################################################

'''
Create a sketch version of the given dataset by randomly sub-sampling.
'''
class SketchDataset(MeshDataset):
    def __init__(self,
            dataset,
            sample_factor,
            sketch_type
        ):

        #hyper-parameters
        self.num_points = dataset.num_points
        self.rank = round(sample_factor*dataset.num_points)
        self.num_snapshots = dataset.num_snapshots

        #Initialize data
        self.points = dataset.points

        #Sketching seeds and sketch features
        #NOTE: I think this isn't ideal, but there are some issues trying to generate seeds other ways
        self.seeds = torch.randint(100000, (self.num_snapshots,))

        self.features = sketch(dataset.features, (self.seeds, self.rank), sketch_type=sketch_type)

        return
    
    def __getitems__(self, idxs):
        if isinstance(idxs, list): idxs = torch.tensor(idxs)

        if idxs.numel() == 0: return None, None, None

        #normalized time
        t_coord = (2*idxs/(self.num_snapshots-1)-1).view(-1,1,1).expand(-1, self.num_points, -1)

        #coordinates
        coordinates = torch.cat((self.points[idxs,:,:], t_coord), dim=2)

        #features
        features = self.features[idxs,:,:]

        return coordinates, features, (self.seeds[idxs], self.rank)

#################################################

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
            gradients = False,
            buffer = None,
            sample_factor = 0.01,
            sketch_type = "fjlt",
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

        #online training
        if self.buffer:
            self.online = True
        else:
            self.online = False

        return
    
    @property
    def input_shape(self):
        return (1, 1, self.spatial_dim+1)

    @property
    def output_shape(self):
        return (1, 1, len(self.channels))

    '''
    Load and preprocess data
    '''
    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=self.gradients)

            if self.online:
                self.train = train_val
                self.sketch = SketchDataset(train_val, sample_factor=self.sample_factor, sketch_type=self.sketch_type) if self.buffer['sketch'] else None

            else:
                train_size = round(self.split*len(train_val))
                val_size = len(train_val) - train_size

                self.train, self.val = random_split(train_val, [train_size, val_size])

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=False)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = MeshDataset(self.points_path, self.features_path, self.channels, normalize=self.normalize, gradients=False)

        if stage not in ["fit", "test", "predict", None]:
            raise ValueError("Stage must be one of fit, test, predict")
        
        return

    '''
    Used in Trainer.fit
    '''
    def train_dataloader(self):
        if self.online:

            loaders = dict()

            if self.buffer.get("full"):
                full_loader = DataLoader(self.train,
                                            batch_sampler=Buffer(self.train.num_snapshots//self.time_span, **self.buffer['full']),
                                            num_workers=self.num_workers*self.trainer.num_devices,
                                            persistent_workers=self.persistent_workers,
                                            pin_memory=self.pin_memory,
                                            collate_fn=lambda x: x)
                
                loaders["full"] = full_loader

            if self.buffer.get("sketch"):
                sketch_loader = DataLoader(self.sketch,
                                            batch_sampler=Buffer(self.train.num_snapshots//self.time_span, **self.buffer['sketch']),
                                            num_workers=self.num_workers*self.trainer.num_devices,
                                            persistent_workers=self.persistent_workers,
                                            pin_memory=self.pin_memory,
                                            collate_fn=lambda x: x)
                
                loaders["sketch"] = sketch_loader

            return CombinedLoader(loaders, mode='max_size_cycle')

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
