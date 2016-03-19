#!/bin/bash
#PBS -N controller
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -q lp
module load compat-openmpi-x86_64
mpirun ipengine &
sleep 24h
