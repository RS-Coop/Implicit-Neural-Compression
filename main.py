'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/config1.yml
    python main.py --experiment config1
'''

from core.model import Model
from core.data import DataModule1
from core.utilities import ProgressBar

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
def main(trainer_args, model_args, data_args, extra_args):

    #Setup data
    data_module = DataModule1(**data_args)

    #Build model
    model = Model1(**model_args)

    #Callbacks
    callbacks = [ProgressBar()]
    if extra_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
    if train_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Train model
    trainer = Trainer(**trainer_args, callbacks=callbacks)
    trainer.fit(model=model, datamodule=data_module, ckpt_path=None) #ckpt_path can be path to partially trained model

    '''
    Do anything else post training here
    '''

'''
Parse arguments
'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    args, _ = parser.parse_known_args()

    #Use command line arguments
    if args == None:
        pass

    #Use YAML configuration file
    else:
        try:
            #open YAML file
            with open(f"experiments/{args.experiment}.yml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            train_args, model_args, data_args, extra_args = config['train'], config['model'], config['data'], config['extra']

        except Exception as e:
            raise ValueError(f"Experiment {args.experiment} is invalid.")

    main(train_args, model_args, data_args, extra_args)
