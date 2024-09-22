'''
DataModule.

LightningDataModule documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html?highlight=DataModule
'''

import pathlib

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.combined_loader import CombinedLoader

from core.datasets import SingleDataset, MultiDataset, SketchSingleDataset, SketchMultiDataset
from core.modules.sampler import Buffer

################################################################################

def get_dataset(features_path, **kwargs):

    #NOTE: This assumes all feature_paths are homogenous
    path_type = pathlib.Path(features_path)

    if path_type.is_file():
        return SingleDataset(features_path=features_path, **kwargs)
    else:
        return MultiDataset(features_path=features_path, **kwargs)
    
def get_sketch_dataset(dataset, **kwargs):

    if isinstance(dataset, SingleDataset):
        return SketchSingleDataset(dataset, **kwargs)
    elif isinstance(dataset, MultiDataset):
        return SketchMultiDataset(dataset, **kwargs)
    else:
        raise ValueError()

################################################################################

'''
'''
class DataModule(LightningDataModule):
    '''
    Input:
        data_dir: path to dataset directory (usually absolute is more robust)
        batch_size: torch dataloader batch size
        num_workers: machine dependent, more workers means faster loading
    '''
    def __init__(self,*,
            spatial_dim,
            points_path,
            features_path,
            batch_size,
            channels,
            time_span,
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

        #Buffer training
        self.online = True if self.buffer else False

        return
    
    @property
    def input_shape(self):
        return (1, self.time_span), (1, 1, 1, self.spatial_dim)

    @property
    def output_shape(self):
        return (1, 1, len(self.channels))

    '''
    Load and preprocess data
    '''
    def setup(self, stage=None):
        if (stage == "fit" or stage is None) and (self.train is None or self.val is None):
            #load dataset
            train_val = get_dataset(self.features_path, points_path=self.points_path, channels=self.channels, time_span=self.time_span, normalize=self.normalize, gradients=self.gradients)

            if self.split != 1.0:
                train_size = round(self.split*len(train_val))
                val_size = len(train_val) - train_size

                self.train, self.val = random_split(train_val, [train_size, val_size])
            else:
                self.train = train_val

            if self.buffer.get("sketch"):
                self.sketch = get_sketch_dataset(self.train, sample_factor=self.sample_factor, sketch_type=self.sketch_type) if self.buffer['sketch'] else None

        if (stage == "test" or stage is None) and self.test is None:
            #load dataset
            self.test = get_dataset(self.features_path, points_path=self.points_path, channels=self.channels, time_span=self.time_span, normalize=self.normalize, gradients=False)

        if (stage == "predict" or stage is None) and self.predict is None:
            #load dataset
            self.predict = get_dataset(self.features_path, points_path=self.points_path, channels=self.channels, time_span=self.time_span, normalize=self.normalize, gradients=False)

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
