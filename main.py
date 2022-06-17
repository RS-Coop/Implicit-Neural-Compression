'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/config1.yml
    python main.py --experiment config1
'''

from core.model import Model
from core.data import DataModule1
from core.utilities import ProgressBar, Logger

import yaml
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

'''
Build and train a model.

Input:
    trainer_args: PT Lightning Trainer arguments
    model_args: model arguments
    data_args: PT LightningDataModule arguments
    extra_args: other arguments that don't fit in groups above
'''
def main(args, trainer_args, model_args, data_args):

    #Setup data
    data_module = DataModule1(**data_args)

    #Build model
    model = Model1(**model_args)

    #Callbacks
    callbacks = []
    if extra_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
    if train_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Logger
    if train_args['logger']:
        train_args['logger'] = Logger(save_dir=train_args['default_root_dir'],
                                        default_hp_metric=False)

    #Train model
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    if train_args['auto_scale_batch_size']:
        trainer.tune(model, datamodule=data_module)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=None) #ckpt_path can be path to partially trained model

    '''
    Do anything else post training here
    '''

'''
Parse arguments
'''
if __name__ == "__main__":
    #Look for CL arguments
    train_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

    #trainer args
    train_parser = Trainer.add_argparse_args(train_parser)

    #parse remaining args
    train_args = vars(train_parser.parse_known_args()[0])

    #Load YAML config
    if args['experiment'] != None:
        try:
            #open YAML file
            with open(f"experiments/{args['experiment']}.yml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            train_args.update(config['train'])
            model_args = config['model']
            data_args = config['data']
            args = config['extra']

        except Exception as e:
            raise ValueError(f"Experiment {args['experiment']} is invalid.")

    main(args, train_args, model_args, data_args)
