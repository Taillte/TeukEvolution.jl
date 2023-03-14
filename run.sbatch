#!/bin/bash
#SBATCH --partition=debugq
#SBATCH --job-name=test

#SBATCH --nodes=1
#SBATCH --output=test.out
#SBATCH --time=0:40:00

set -euxo pipefail

pwd
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
date

module load julia/1.6.0

env OMP_NUM_THREADS=40 julia ./examples/run3.jl

date

