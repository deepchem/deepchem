#!/bin/bash
#SBATCH --job-name=arrayJob
#SBATCH --output=arrayJob_%A_%a.out
#SBATCH --error=arrayJob_%A_%a.err
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH --partition=owners
srun ipengine
