'''
Trains a model.
'''

import os
from importlib import import_module

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profilers import AdvancedProfiler

from core.model import Model
from core.data import DataModule

from .utils.utils import Logger, FineTuner

def train(config_path, config):

    #Extract args
    data_args = config['data']
    model_args = config['model']
    trainer_args = config['train']
    misc_args = config['misc']

    #Setup datamodule
    datamodule = DataModule(**data_args)

    #Build model
    if 'ckpt_path' in model_args.keys():
        model = Model.load_from_checkpoint(model_args.pop('ckpt_path'), input_shape=datamodule.input_shape, output_shape=datamodule.output_shape, **model_args)
    else:
        model = Model(input_shape=datamodule.input_shape, output_shape=datamodule.output_shape, **model_args)

    #Torch Compile
    model = torch.compile(model)

    #Callbacks
    callbacks=[]

    if misc_args['early_stopping']:
        callbacks.append(EarlyStopping(monitor="val_err"))

    if trainer_args['enable_checkpointing']:
        if trainer_args.get('limit_val_batches') != 0:
            monitor = 'val_err'
            k = 1
        else:
            monitor = None
            k = 0

        callbacks.append(ModelCheckpoint(monitor=monitor,
                                            save_last=True,
                                            save_top_k=k,
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
        
    #Profiler
    if trainer_args.get("profiler") == "advanced":
        trainer_args['profiler'] = AdvancedProfiler(dirpath=logger.log_dir, filename="perf")

    #Fine tuner
    if misc_args.get('finetune'):
        callbacks.append(FineTuner())

    #Build trainer
    trainer = Trainer(**trainer_args, callbacks=callbacks)

    #Train model
    trainer.fit(model=model, datamodule=datamodule)

    return
