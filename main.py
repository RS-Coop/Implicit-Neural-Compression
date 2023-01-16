'''
Builds, trains, and tests a model based on input parameters, which are specified
via a YAML configuration file and optional command line arguments.

Example usage:
    - Run the test found in experiments/tests/config1.yaml
        python main.py --experiment tests/config1.yaml
'''

import os
import yaml
from argparse import ArgumentParser

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.model import Model
from core.data import DataModule
from core.utils import ProgressBar, Logger

'''
Build, train, and test a model.

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
    if "ckpt_path" in model_args.keys():
        ckpt_path = model_args["ckpt_path"]
        model = None
    else:
        ckpt_path = None
        model = Model(**model_args)

    #Callbacks
    callbacks = []
    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping

    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Logger
    if trainer_args['logger']:
        #Save config details
        exp_dir, exp_name = os.path.split(experiment)
        exp_name = os.path.splitext(exp_name)[0]

        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], exp_dir),
                        name=exp_name, default_hp_metric=False)

        logger.log_config(config)

        #add logger to trainer args
        trainer_args['logger'] = logger

    #Train model
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    if trainer_args['auto_scale_batch_size']:
        trainer.tune(model, datamodule=data_module)

    trainer.fit(model=model, datamodule=data_module, ckpt_path=ckpt_path)

    #Compute testing statistics
    if misc_args['compute_stats']:
        trainer.test(model=None if trainer_args['enable_checkpointing'] else model,
                        ckpt_path='best' if trainer_args['enable_checkpointing'] else None,
                        datamodule=datamodule)

    '''
    Do anything else post training here
    '''

'''
Parse arguments from configuration file and command line. Only Lightning Trainer
command line arguments will override their config file counterparts.
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

    #look for other CL arguments
    config['train'].update(vars(trainer_parser.parse_known_args()[0]))

    #run main script
    main(experiment, config)
