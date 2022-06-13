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
