#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH -A quantum
#SBATCH --partition=largemem
#SBATCH -N 1     # nodes
#SBATCH -c 45     # cpus-per-task ("cores")
#SBATCH --output=slurm-output/training-%A.log
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
# target, reward
time python train.py 45 $1 $2
