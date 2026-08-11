#!/bin/bash
#SBATCH --job-name=bench-mc
#SBATCH --partition=compute
#SBATCH --chdir=/home/hideaki/Dropbox/lab/github/ssll
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/bin/slurm-preamble.sh ssll

# Let the numba kernels in boltzmann_learning use the allocated cores
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Usage: sbatch submit_bench_mc.sh <N> [T] [R] [EM_ITERS]
python test/bench_mc_largeN.py "$@"
