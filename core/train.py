'''
Trains a model.
'''

import os
from importlib import import_module

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from core.model import Model
from core.data import DataModule
from .utils import Logger

def train(config_path, config):

    #Extract args
    data_args = config['data']
    model_args = config['model']
    trainer_args = config['train']
    misc_args = config['misc']

    #Setup datamodule
    data_module = DataModule(**data_args)

    #Build model
    if 'ckpt_path' in model_args.keys():
        model = Model.load_from_checkpoint(model_args.pop('ckpt_path'), **model_args)
    else:
        model = Model(**model_args)

    #Callbacks
    callbacks=[]
    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_loss"))

    if trainer_args['enable_checkpointing']:
        callbacks.append(ModelCheckpoint(monitor='val_loss',
                                            save_last=True,
                                            save_top_k=1,
                                            mode='min',
                                            filename='best-{epoch}'))

    #Logger
    if trainer_args['logger']:

        #Save config details
        logger = Logger(save_dir=os.path.join(trainer_args['default_root_dir'], config_path),
                        name='', default_hp_metric=False)

        logger.log_config(config)

        #Add logger to trainer args
        trainer_args['logger'] = logger

    #Build trainer
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    #Train model
    trainer.fit(model=model, datamodule=datamodule)

    #Compute testing statistics
    if misc_args['compute_stats']:
        trainer.test(model=None if trainer_args['enable_checkpointing'] else model,
                        ckpt_path='best' if trainer_args['enable_checkpointing'] else None,
                        datamodule=datamodule)

    '''
    Do anything else post training here
    '''

    return
