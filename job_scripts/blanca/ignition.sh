#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=ignition
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue

#ntasks-per-node should match num_gpus
#cpus-per-task per node should be num_workers per gpu

TEST=ignition/hnet_online_baseline
TIME=00:01:00:00
PYTHON=/projects/cosi1728/software/anaconda/envs/compression/bin/python

DATA_DIR=data/ignition
cp -r $DATA_DIR/points.npy $SLURM_SCRATCH
cp -r $DATA_DIR/features.npy $SLURM_SCRATCH

module purge
module load anaconda

conda activate compression

srun $PYTHON run.py --mode train --config $TEST --max_time $TIME --data_dir $SLURM_SCRATCH
srun $PYTHON run.py --mode test --config $TEST/version_0 --data_dir $SLURM_SCRATCH