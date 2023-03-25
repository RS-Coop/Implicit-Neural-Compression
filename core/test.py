'''
Tests a model.
'''

from pathlib import Path

import torch
from pytorch_lightning import Trainer

from core.model import Model
from core.data import DataModule
from .utils import Logger

def test(log_dir, config):

    #Extract args
    data_args = config['data']
    model_args = config['model']
    trainer_args = config['train']

    #Setup datamodule
    datamodule = DataModule(**data_args)

    #Build model
    try:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('best-epoch=*.ckpt'))[0]
    except:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('last.ckpt'))[0]

    model = Model.load_from_checkpoint(ckpt_path, **model_args)

    #Build trainer
    logger = Logger(save_dir=log_dir, name='', version='', default_hp_metric=False)

    trainer_args['logger'] = logger
    trainer_args["devices"] = 1

    trainer = Trainer(**trainer_args, inference_mode=True)

    #compute testing statistics
    trainer.test(model=model, datamodule=datamodule)
