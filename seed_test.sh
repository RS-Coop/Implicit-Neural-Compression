#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --job-name=seed_test
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue

#ntasks-per-node should match num_gpus
#cpus-per-task per node should be num_workers per gpu

PYTHON=/projects/cosi1728/software/anaconda/envs/compression/bin/python

srun $PYTHON seed_test.py