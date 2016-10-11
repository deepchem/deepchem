#!/bin/bash
#PBS -N controller
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
##PBS -q lp
module load compat-openmpi-x86_64 
for i in {0..24}; do
    echo $i;
    mpirun -np 1 --bind-to-socket ipengine &
    sleep 5s
done
sleep 24h
