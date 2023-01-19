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
    trainer_args = config['train']
    model_args = config['model']
    data_args = config['data']

    #Build trainer
    logger = Logger(save_dir=log_dir, name='', version='', default_hp_metric=False)

    trainer_args['logger'] = logger
    trainer_args["devices"] = 1
    # trainer_args['inference_mode'] = True

    trainer = Trainer(**trainer_args)

    #Setup datamodule
    datamodule = DataModule(**data_args)

    #Build model
    ckpt_path = list(Path(log_dir, 'checkpoints').rglob('best-epoch=*.ckpt'))[0]

    model = Model.load_from_checkpoint(ckpt_path, **model_args)

    with torch.inference_mode():
        #compute testing statistics
        trainer.test(model=model, datamodule=datamodule)
