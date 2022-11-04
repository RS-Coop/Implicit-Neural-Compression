# How to use
This is meant to be used as a template for a PyTorch and PT Lightning based project. The structure is what I find most convenient, and it should contain everything you need to get started. Feel free to rename, rearrange, add, and remove anything you want. Let me know if you have something I should add, or if an existing part of the template is incorrect.

The file `environment.yml` can be used to build a conda environment as follows:
```bash
conda env create -f environment.yaml
```
which will install all necessary packages for this template.

For local development run the following command from within the top level of this repository:
```bash
pip install -e .
```
This will install *core* as a pip package in editable mode so that local changes are automatically updated.

To run an experiment:
```bash
python main.py --experiment <path to YAML file within ./experiments>
```

To visualize the logging results saved to `lightning_logs/` run the following command:
```bash
tensorboard --logdir=lightning_logs/
```

## Tips and Tricks
- Don't use ```.cuda()``` or ```.to(device)```, PT Lightning should handle all of this internally, and in the rare case where you do need to place a tensor on the correct device yourself, you should do this in an agnostic manner.
- PT Lightning has a lot of functionality, so always check the documentation.

## Structure
- *core*: Model architectures, data loading, utilities, and core operators
- *data*: Data folders
- *experiments*: Experiment configuration files
  - *template.yaml*: Detailed experiment template
- *job_scripts*: HPC job submission scripts
- *lightning_logs*: Experiment logs
- *notebooks*: Various Jupyter Notebooks
- *py_scripts*: Various python scripts
- *main.py*: Model training and testing script

## Documentation
There is a lot of documentation available, but I have picked out some of what I think is the most useful.

- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/): Documentation homepage
- [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule): Overview and introduction on how to use
- [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html): Overview and introduction on how to use
- [Logging](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html): How to log metrics
