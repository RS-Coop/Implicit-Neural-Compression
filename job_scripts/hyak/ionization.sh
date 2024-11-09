#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=ionization
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --no-requeue

#ntasks-per-node should match num_gpus
#cpus-per-task per node should be num_workers per gpu

TEST=ionization/hnet_online_subsample_1%
TIME=00:23:00:00
PYTHON=/projects/cosi1728/software/anaconda/envs/compression/bin/python

DATA_DIR=data/ionization
cp -r $DATA_DIR/points_0.npy $SLURM_SCRATCH/points.npy
cp -r $DATA_DIR/features_0 $SLURM_SCRATCH/features

module purge
module load anaconda

conda activate compression

srun $PYTHON run.py --mode train --config $TEST --max_time $TIME --data_dir $SLURM_SCRATCH
srun $PYTHON run.py --mode test --config $TEST/version_0 --data_dir $SLURM_SCRATCH
