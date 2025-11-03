# *In Situ* Training of Implicit Neural Compressors for Scientific Simulations via Sketch-Based Regularization

### [arXiv]()

[Cooper Simpson](https://rs-coop.github.io/), [Stephen Becker](https://stephenbeckr.github.io/), [Alireza Doostan](https://www.colorado.edu/lab/uq-data-driven-modeling/)

Submitted to [Journal of Computational Physics](https://www.sciencedirect.com/journal/journal-of-computational-physics)

## Abstract

Focusing on implicit neural representations, we present a novel *in situ* training protocol that employs limited memory buffers of full and sketched data samples, where the sketched data are leveraged to prevent catastrophic forgetting. The theoretical motivation for our use of sketching as a regularizer is presented via a simple Johnson-Lindenstrauss-informed result. While our methods may be of wider interest in the field of *continual learning*, we specifically target *in situ* neural compression using implicit neural representation-based hypernetworks. We evaluate our method on a variety of complex simulation data in two and three dimensions, over long time horizons, and across unstructured grids and non-Cartesian geometries. On these tasks, we show strong reconstruction performance at high compression rates. Most importantly, we demonstrate that sketching enables the presented *in situ* scheme to approximately match the performance of the equivalent offline method.

## License \& Citation
All source code is made available under an MIT license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See [`LICENSE`](./LICENSE) for the full text. 

Our work can be cited using the following bibtex entry:
```bibtex
@article{simpson2025inc,
  title = {{Sketch-Based Online Training of Implicit Neural Compressors for Scientific Simulations}},
  authors = {Simpson, Cooper and Becker, Stephen and Doostan, Alireza},
  year = {2025},
  journal = {arXiv}
}
```

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
conda env create --file=environment.yaml --name=compression
```
which will install all necessary packages for this template in the conda environment `compression`.

For local development, and if you want to be able to run the notebooks, it is easiest to install `core` as a pip package in editable mode using the following command from within the top level of this repository:
```console
pip install -e .
```
Although, the main experiment script can still be run without doing this.

### Data Acquisition
The *Ignition* and *Neuron Transport* datasets are not publically available, and the *Channel Flow* dataset is a trimmed variant of the full version from the [JHU Turbulence Database](https://turbulence.idies.jhu.edu/datasets/wallBoundedTurbulence/channelFlow). **Please** reach out if you would like access or further information.

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
The Jupyter Notebook [`Paper Results`](./Paper%20Results.ipynb) can be used to generate all paper figures and table details.
