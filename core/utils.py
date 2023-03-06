'''
Extra utilities.

e.g. custom loss function
'''

from pytorch_lightning.callbacks.progress import TQDMProgressBar

'''
Custom PT Lightning training progress bar.

Documentation:
    https://pytorch-lightning.readthedocs.io/en/stable/common/progress_bar.html
'''
class ProgressBar(TQDMProgressBar):
    def __init__(self):
        super().__init__()

'''
Custom Tensorboard logger.
'''
class Logger(TensorBoardLogger):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    '''
    Dont log epoch metric
    '''
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    '''
    Dont log hparams.yaml
    '''
    @rank_zero_only
    def save(self):
        pass

    '''
    Log configuration file
    '''
    @rank_zero_only
    def log_config(self, config):

        filename = os.path.join(self.log_dir, 'config.yaml')
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w") as file:
            yaml.dump(config, file)

        return

'''
Fine tuning callback to freeze particular layers before training.
'''
class FineTuner(BaseFinetuning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def freeze_before_training(self, pl_module):
        pass

    def finetune_function(self, pl_module, current_epoch, optimizer, optimizer_idx):
        pass
