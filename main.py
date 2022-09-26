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

import os
import yaml
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

'''
Build and train a model.

Input:
    experiment: experiment name
    trainer_args: PT Lightning Trainer arguments
    model_args: model arguments
    data_args: PT LightningDataModule arguments
    extra_args: other arguments that don't fit in groups above
'''
def main(experiment, trainer_args, model_args, data_args, extra_args):

    #Setup data
    data_module = DataModule1(**data_args)

    #Build model
    model = Model1(**model_args)

    #Callbacks
    callbacks = []
    if extra_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor='val_loss')) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping
    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint()) #https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.ModelCheckpoint.html

    #Logger
    if trainer_args['logger']:
        logger = Logger(save_dir=trainer_args['default_root_dir'],
                                        name=experiment,
                                        default_hp_metric=False)

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
Parse arguments
'''
if __name__ == "__main__":
    #Look for config
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    args = vars(parser.parse_known_args()[0])

    #Load YAML config
    if args['experiment'] != None:
        try:
            #open YAML file
            with open(f"experiments/{args['experiment']}.yaml", "r") as file:
                config = yaml.safe_load(file)

            #extract args
            trainer_args = config['train']
            model_args = config['model']
            data_args = config['data']
            extra_args = config['extra']

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
    model_parser = AutoEncoder.add_args(model_parser)

    #data specific args
    data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    data_parser = GridDataModule.add_args(data_parser)

    #look for other CL arguments
    trainer_args.update(vars(trainer_parser.parse_known_args()[0]))
    model_args.update(vars(model_parser.parse_known_args()[0]))
    data_args.update(vars(data_parser.parse_known_args()[0]))

    #run main script
    main(trainer_args, model_args, data_args, extra_args)
