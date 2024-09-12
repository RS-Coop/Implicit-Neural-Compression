# Sketching Based Online Training of Implicit Neural Compressors for Scientific Simulations

### [arXiv]()

[Cooper Simpson](https://rs-coop.github.io/), [Stephen Becker](https://stephenbeckr.github.io/), [Alireza Doostan](https://www.colorado.edu/aerospace/alireza-doostan)

Submitted to []()

## Abstract

## License \& Citation
All source cod is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See [`LICENSE`](./LICENSE) for the full text. 

Our work can be cited using the Bibtex entry in [`CITATION`](./CITATION).

## Reproducibility

### Repository Structure
- [`core`](./core/): Model architecture, data loading, utilities, and any core operations
- [`data`](./data/): Data folders
- [`experiments`](./experiments/): Experiment configuration files
  - [`template.yaml`](./experiments/template.yaml): Detailed experiment template
- [`lightning_logs`](./lightning_logs/): Experiment logs
- [`job_scripts`](./job_scripts/): SLURM job scripts
- [`run.py`](./run.py): Model training and testing script

### Environment Setup
The file `environment.yaml` contains a list of dependencies, and it can be used to generate an anaconda environment with the following command:
```console
conda create -f=environment.yaml -n=compression
```
which will install all necessary packages for this template in the conda environment `compression`.

For local development, and if you want to be able to run the notebooks, it is easiest to install `core` as a pip package in editable mode using the following command from within the top level of this repository:
```console
pip install -e .
```
Although, the main experiment script can still be run without doing this.

### Data Acquisition
The *Ignition*, *Neuron Transport*, and *Channel Flow* datasets used in our paper can be downloaded using Git LFS using the following command:
```console
...
```
Note that *Channel Flow* comes from the [JHU Turbulence Database](https://turbulence.idies.jhu.edu/datasets/wallBoundedTurbulence/channelFlow), but we use a trimmed variant.

The *Ionization* dataset can be accessed at the [IEEE Visualization 2008 Design Contest](https://sciviscontest.ieeevis.org/2008/data.html) website. We use the second channel, temperature, and the first 100 snapshots of the available 200. Download the data into [`data/ionization/`](./data/ionization/), unzip, and use the script [`ionization_convert.py`](./data/ionization/ionization_convert.py) to convert the unzipped snapshot files into `.npy` files and generate an associated `points.npy` file.

### Running Experiments
Use the following command to run an experiment:
```console
python run.py --mode train --config <path to YAML file within ./experiments> --data_dir <path to data>
```
If `logger` is set to `True` in the YAML config file, then the results of this experiment will be saved to `lightning_logs/<path to YAML file within ./experiments>`, and use this command to test a logged run:
```console
python run.py --mode test --config <path to version inside ./lightning_logs> --data_dir <path to data>
```
To visualize the logging results saved to `lightning_logs/` using tensorboard run the following command:
```console
tensorboard --logdir=lightning_logs/
```

### Generating Figures
The Jupyter Notebook [`Paper Figures`](./Paper%20Figures.ipynb) can be used to generate all paper figures and tables.