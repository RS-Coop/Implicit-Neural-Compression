#!/bin/bash
#SBATCH --job-name=dimension-analysis
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#NOTE: Memory should be total memory for all cpus
#NOTE: ntasks-per-node should match num_gpus
#NOTE: cpus-per-task per node should be num_workers per gpu

module purge
source ~/.bashrc #For conda

conda activate compression

python dimension_analysis.py