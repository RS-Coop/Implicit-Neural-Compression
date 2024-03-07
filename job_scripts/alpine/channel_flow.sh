#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=channel_flow
#SBATCH --qos=normal
#SBATCH --partition=aa100
#SBATCH --account=ucb332_asc1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --no-requeue

#ntasks-per-node should match num_gpus
#cpus-per-task per node should be num_workers per gpu

TEST=channel_flow/siren
TIME=00:13:00:00
PYTHON=/projects/cosi1728/software/anaconda/envs/compression/bin/python

DATA_DIR=data/channel_flow
cp -r $DATA_DIR/points_64.npy $SLURM_SCRATCH
cp -r $DATA_DIR/features_64_1000.npy $SLURM_SCRATCH

module purge
module load anaconda

conda activate compression

srun $PYTHON run.py --mode train --config $TEST --max_time $TIME --data_dir $SLURM_SCRATCH
#srun $PYTHON run.py --mode test --config $TEST/version_0 --data_dir $SLURM_SCRATCH