#!/bin/bash
#PBS -N controller
##PBS -l select=1:ncpus=1:mem=1gb
#PBS -l nodes=1:ppn=2
#PBS -l walltime=24:00:00
##SBATCH --output=controller_%A_%a.out
##SBATCH --error=controller_%A_%a.err
##SBATCH --time=48:00:00
##SBATCH --partition=normal
##SBATCH --qos=bigmem
##SBATCH -n 1
ipcontroller --ip='*' --log-to-file=True &
#bash run_engines.sh
#for i in {0..150}; do
#    echo $i;
#    qsub engine.sh &
#done
sleep 24h
