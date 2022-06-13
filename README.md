# How to use
This is meant to be used as a template for a PyTorch and PT Lightning based project. The structure is what I find most convenient, and it should contain everything you need to get started. Feel free to rename, rearrange, add, and remove anything you want. Let me know if you have something I should add, or if an existing part of the template is incorrect.

The file `environment.yml` can be used to build a conda environment as follows:
```bash
conda env create -f environemnt.yml
```
which will install all necessary packages for this template.

## Tips and Tricks
- Don't use ```.cuda()``` or ```.to(device)```, PT Lightning should handle all of this internally, and in the rare case where you do need to place a tensor on the correct device yourself, you should do this in an agnostic manner.
- PT Lightning has a lot of functionality, so always check the documentation.

## Documentation
There is a lot of good documentation (and some not so good) available, but I have picked out some of what I think is the most useful.

- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/): Documentation homepage
- [LightningModule](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.LightningModule.html?highlight=LightningModule): Overview and introduction on how to use
- [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html): Overview and introduction on how to use
- [Logging](https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html): How to log metrics
