'''
Tests a model.
'''

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import Trainer

from .gif import make_gif

from core.model import Model
from core.data import DataModule
# from core.INC import Model
# from core.data_2 import DataModule
from .utils import Logger

def test(log_dir, config):

    #Extract args
    data_args = config['data']
    model_args = config['model']
    trainer_args = config['train']
    misc_args = config['misc']

    #Setup datamodule
    datamodule = DataModule(**data_args)

    #Build model
    try:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('best-epoch=*.ckpt'))[0]
    except:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('last.ckpt'))[0]

    if 'ckpt_path' in model_args.keys():
        model_args.pop('ckpt_path')

    model = Model.load_from_checkpoint(ckpt_path, input_shape=datamodule.input_shape, output_shape=datamodule.output_shape, **model_args)

    #Build trainer
    logger = Logger(save_dir=log_dir, name='', version='', default_hp_metric=False)

    trainer_args['logger'] = logger
    trainer_args["devices"] = 1

    if "strategy" in trainer_args.keys():
        trainer_args.pop("strategy")

    trainer = Trainer(**trainer_args, inference_mode=True)

    #compute testing statistics
    trainer.test(model=model, datamodule=datamodule)

    #make GIF
    if misc_args.get('make_gif'):
        make_gif(trainer, datamodule, model)

    if misc_args.get('compute_stats'):
        #run on test
        datamodule.setup('test')
        model.prefix = 'real_'
        model.denormalize = datamodule.test.denorm_f

        with torch.inference_mode():
            trainer.test(model=model, datamodule=datamodule)

    if misc_args.get('export_txt'):
        with torch.inference_mode():
            data = trainer.predict(model=model, datamodule=datamodule)

        data = datamodule.test.denorm_f(torch.stack(data))
        
        np.save(f'{trainer.logger.log_dir}/reconstruction.npy', data.numpy())

    return

