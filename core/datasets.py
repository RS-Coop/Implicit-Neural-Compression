'''
Data loading utilities.

LightningDataModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html?highlight=DataModule
'''

import pathlib
from natsort import natsorted
from warnings import warn

import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils.sketch import sketch

'''
'''
class SingleDataset(Dataset):
    def __init__(self,*,
            points_path,
            features_path,
            channels,
            time_span,
            channels_last = True,
            normalize = True,
            gradients = False
        ):
        super().__init__()

        try:
            #Features
            features = torch.from_numpy(np.load(features_path).astype(np.float32))

            assert features.dim() == 3, f"Features has {features.dim()} dimensions, but should have 3"

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

        if self.num_snapshots%time_span != 0: warn("Number of training snapshots not evenly divisible by time span!")
        self.time_span = time_span

        return
    
    @property
    def size(self):
        return self.features.numel()

    def __len__(self):
        return self.num_snapshots//self.time_span
    
    def __getitems__(self, idxs):
        if isinstance(idxs, list): idxs = torch.tensor(idxs)

        if idxs.numel() == 0: return None, None

        #Convert window idxs to snapshot idxs
        idxs = torch.stack([torch.arange(idx*self.time_span, (idx+1)*self.time_span) for idx in idxs])

        #Normalized time
        t_coord = (2*idxs/(self.num_snapshots-1)-1) if self.num_snapshots != 0 else 0.0*idxs

        #Coordinates
        x_coord = self.points[idxs,:,:]

        #Features
        if self.gradients != None:
            features = torch.cat((self.features[idxs,:,:], torch.flatten(self.gradients[idxs,...], start_dim=2)), dim=2)
        else:
            features = self.features[idxs,:,:]

        return (t_coord, x_coord), torch.flatten(features, start_dim=0, end_dim=1)
    
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
class SketchSingleDataset(SingleDataset):
    def __init__(self,
            dataset,
            sample_factor,
            sketch_type
        ):

        #hyper-parameters
        self.num_points = dataset.num_points
        self.num_snapshots = dataset.num_snapshots
        self.time_span = dataset.time_span
        self.rank = round(sample_factor*dataset.num_points)

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

        #Convert window idxs to snapshot idxs
        idxs = torch.stack([torch.arange(idx*self.time_span, (idx+1)*self.time_span) for idx in idxs])

        #Normalized time
        t_coord = (2*idxs/(self.num_snapshots-1)-1) if self.num_snapshots != 0 else 0.0*idxs

        #Coordinates
        x_coord = self.points[idxs,:,:]

        #Features
        features = self.features[idxs,:,:]

        return (t_coord, x_coord), torch.flatten(features, start_dim=0, end_dim=1), (self.seeds[idxs], self.rank)
    
#################################################

'''
'''
class MultiDataset(Dataset):
    def __init__(self,*,
            points_path,
            features_path,
            channels,
            time_span,
            channels_last = True,
            normalize = True,
            gradients = False
        ):
        super().__init__()

        #Set attributes
        self.channels = channels
        self.channels_last = channels_last

        try:
            #Features

            #Get snapshot files
            self.snapshot_files = natsorted(pathlib.Path(features_path).glob("*"))

            if len(self.snapshot_files) == 0: raise Exception(f'No features have been found in: {features_path}')

            if gradients:
                raise NotImplementedError("Gradients not implemented for multi-file dataset.")
            else:
                self.gradients = None

            #Points
            points = torch.from_numpy(np.load(points_path).astype(np.float32))

            if points.dim() == 2:
                self.points = points.expand(len(self.snapshot_files), -1, -1)
            elif points.dim() == 3:
                self.points = points
            else:
                raise Exception(f'Points has incorrect number of dimensions')

        except FileNotFoundError:
            raise Exception(f'Error loading points {points_path} and/or features {features_path}')

        except Exception as e:
            raise e
        
        #Normalize points
        mx = torch.amax(self.points, dim=(0,1))
        mi = torch.amin(self.points, dim=(0,1))
        self.points = 2*(self.points-mi)/(mx-mi)-1

        self.denorm_p = lambda p: ((p+1)/2)*(mx-mi).to(p.device) + mi.to(p.device)
        
        #Normalize features
        if normalize != False:
            raise NotImplementedError("Normalization not implemented for multi-file dataset.")
        
        self.denorm_f = lambda f: f

        #
        self.num_points = self.points.shape[1]
        self.num_snapshots = len(self.snapshot_files)

        if self.num_snapshots%time_span != 0: warn("Number of training snapshots not evenly divisible by time span!")
        self.time_span = time_span

        return
    
    @property
    def size(self):
        return self.num_snapshots*self.num_points*len(self.channels)

    def __len__(self):
        return self.num_snapshots//self.time_span
    
    def _loaditem(self, idx):

        file = self.snapshot_files[idx]

        if file.suffix == ".npy":
            features = np.load(file).astype(np.float32)
        else:
            features = np.loadtxt(file).astype(np.float32)

        features = torch.from_numpy(features)

        #NOTE: Changing this just for the ionization data
        # assert features.dim() == 2, f"Features has {features.dim()} dimensions, but should only have 2"
        assert features.dim() == 1, f"Features has {features.dim()} dimensions, but should only have 2"
        features = features.unsqueeze(dim=1)

        #Move to channels last
        if self.channels_last == False:
            features = torch.movedim(features, 0, 1)

        #Extract channels
        features = features[:,self.channels]

        return features
    
    def __getitems__(self, idxs):
        if isinstance(idxs, list): idxs = torch.tensor(idxs)

        if idxs.numel() == 0: return None, None

        #Convert window idxs to snapshot idxs
        idxs = torch.stack([torch.arange(idx*self.time_span, (idx+1)*self.time_span) for idx in idxs])

        #Normalized time
        t_coord = (2*idxs/(self.num_snapshots-1)-1) if self.num_snapshots != 0 else 0.0*idxs

        #Coordinates
        x_coord = self.points[idxs,:,:]

        #Features
        features = torch.stack([self._loaditem(i) for i in idxs])

        return (t_coord, x_coord), features
    
    def get_points(self, denormalize=True):

        if denormalize:
            points = self.denorm_p(self.points)
        else:
            points = self.points

        return points[0,:,:] #NOTE: Assuming points are static across time
    
    def get_features(self, denormalize=True):

        if denormalize:
            warn("Features were never normalized to begin with.")

        return torch.stack([self._loaditem(i) for i in range(self.num_snapshots)])
    
#################################################

'''
Create a sketch version of the given dataset by randomly sub-sampling.
'''
class SketchMultiDataset(MultiDataset):
    def __init__(self,
            dataset,
            sample_factor,
            sketch_type
        ):

        #hyper-parameters
        self.dataset = dataset
        self.rank = round(sample_factor*dataset.num_points)
        self.sketch_type = sketch_type

        #Sketching seeds and sketch features
        #NOTE: I think this isn't ideal, but there are some issues trying to generate seeds other ways
        self.seeds = torch.randint(100000, (self.dataset.num_snapshots,))

        return
    
    def _loaditem(self, idx):
        features = self.dataset._loaditem(idx)

        return sketch(features.expand(1,-1,-1), (self.seeds[idx], self.rank), sketch_type=self.sketch_type)
    
    def __getitems__(self, idxs):
        if isinstance(idxs, list): idxs = torch.tensor(idxs)

        if idxs.numel() == 0: return None, None, None

        #Convert window idxs to snapshot idxs
        idxs = torch.stack([torch.arange(idx*self.dataset.time_span, (idx+1)*self.dataset.time_span) for idx in idxs])

        #Normalized time
        t_coord = (2*idxs/(self.dataset.num_snapshots-1)-1) if self.dataset.num_snapshots != 0 else 0.0*idxs

        #Coordinates
        x_coord = self.dataset.points[idxs,:,:]

        #Features
        features = torch.cat([self._loaditem(i) for i in idxs])

        return (t_coord, x_coord), features, (self.seeds[idxs], self.rank)