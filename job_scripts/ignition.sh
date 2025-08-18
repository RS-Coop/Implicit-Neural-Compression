#!/bin/bash
#SBATCH --job-name=ignition
#SBATCH --account=amath
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=08:00:00

#NOTE: ntasks-per-node should match num_gpus
#NOTE: cpus-per-task per node should be num_workers per gpu

TEST=ignition/hnet_online_fjlt_0.7%
DATA_DIR=/gscratch/amath/cooper/data/ignition
TIME=00:07:30:00

module purge
source ~/.bashrc #For conda

conda activate compression

python run.py --mode train --config $TEST --max_time $TIME --data_dir $DATA_DIR
python run.py --mode test --config $TEST/version_0 --data_dir $DATA_DIR
