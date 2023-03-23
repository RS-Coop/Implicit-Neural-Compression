# How to use
This is meant to be used as a template for a PyTorch and Lightning based project. The structure is what I find most convenient, and it should contain everything you need to get started. Feel free to rename, rearrange, add, and remove anything you want. Let me know if you have something I should add, or if an existing part of the template is incorrect.

## Environment Setup
The file `environment.yaml` contains a list of dependencies, and it can be used to generate an anaconda environment with the following command:
```console
conda create -f=environment.yaml -n=<environment-name>
```
which will install all necessary packages for this template in the conda environment `environment-name`.

For local development, it is easiest to install `core` as a pip package in editable mode using the following command from within the top level of this repository:
```console
pip install -e .
```
Although, the main experiment script can still be run without doing this.

## Running Experiments
Use the following command to run an experiment:
```console
python main.py --experiment <path to YAML file within ./experiments>
```
If `logger` is set to `True` in the YAML config file, then the results of this experiment will be saved to `lightning_logs/<path to YAML file within ./experiments>`.

To visualize the logging results saved to `lightning_logs/` using tensorboard run the following command:
```console
tensorboard --logdir=lightning_logs/
```

## Tips and Tricks
- Don't use `.cuda()` or `.to(device)`
- In the rare case where you do need to place a tensor on the correct device yourself, you should do this in an agnostic manner
- If you need parameters to be placed on the correct device by Lightning, make sure they are registered as `torch.nn.Parameter`
- Lightning has a lot of functionality, so always check the documentation.

## Structure
- `core`: Model architectures, data loading, utilities, and core operators
- `data`: Data folders
- `experiments`: Experiment configuration files
  - `template.yaml`: Detailed experiment template
- `lightning_logs`: Experiment logs
- `main.py`: Model training and testing script

## Documentation
There is a lot of documentation available, but I have picked out some of what I think is the most useful.

- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/): Documentation homepage
- [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule): Overview and introduction on how to use
- [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html): Overview and introduction on how to use
- [Logging](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html): How to log metrics
