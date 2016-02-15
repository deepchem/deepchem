#!/bin/bash
#SBATCH --job-name=controller
#SBATCH --output=controller_%A_%a.out
#SBATCH --error=controller_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH -n 1
ipcontroller --ip='*' --log-to-file=False &
for i in {0..90}; do
    echo $i;
    sbatch engine.sh
done
sleep 48h
