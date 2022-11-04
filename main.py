'''
Builds and trains a model based on input parameters, which can be specified via
command line arguments or an experiment YAML file.

Example usage:

- Run the test found in experiments/tests/config1.yaml
    python main.py --experiment tests/config1.yaml
'''

import os
import yaml
from argparse import ArgumentParser
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.model import Model
from core.data import DataModule
from core.utilities import ProgressBar, Logger

'''
Build and train a model.

Input:
    experiment: experiment name
    config: experiment configuration
'''
def main(experiment, config):

    #Extract args
    trainer_args = config['train']
    model_args = config['model']
    data_args = config['data']
    misc_args = config['misc']

    #Setup data
    data_module = DataModule(**data_args)

    #Build model
    model = Model(**model_args)

    #Callbacks
    callbacks = []
    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Logger
    if trainer_args['logger']:
        #Sace config details
        exp_dir, exp_name = os.path.split(experiment)
        exp_name = os.path.splitext(exp_name)[0]

        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], exp_dir),
                        name=exp_name, default_hp_metric=False)

        filename = os.path.join(logger.log_dir, 'config.yaml')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            yaml.dump(config, file)

        #add logger to trainer args
        trainer_args['logger'] = logger

    #Train model
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    if trainer_args['auto_scale_batch_size']:
        trainer.tune(model, datamodule=data_module)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=None) #ckpt_path can be path to partially trained model

    '''
    Do anything else post training here
    '''

'''
Parse arguments from configuration file and command line.
Command line arguments will override their config file counterparts.
'''
if __name__ == "__main__":
    #Look for config
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    experiment = vars(parser.parse_known_args()[0])['experiment']

    #Load YAML config
    if experiment != None:
        try:
            #open YAML file
            with open(f"experiments/{args['experiment']}.yaml", "r") as file:
                config = yaml.safe_load(file)

        except Exception as e:
            raise ValueError(f"Experiment {args['experiment']} is invalid.")
    else:
        raise ValueError("An experiment configuration file must be provided.")

    #trainer args
    trainer_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    trainer_parser.add_argument("--default_root_dir", type=str)
    trainer_parser.add_argument("--max_time", type=str)

    #model specific args
    model_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    model_parser = Model.add_args(model_parser)

    #data specific args
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = DataModule.add_args(data_parser)

    #look for other CL arguments
    config['train'].update(vars(trainer_parser.parse_known_args()[0]))
    config['model'].update(vars(model_parser.parse_known_args()[0]))
    config['data'].update(vars(data_parser.parse_known_args()[0]))

    #run main script
    main(experiment, config)
