#!/bin/bash
#SBATCH --time=7-00:00:00
#SBATCH -A quantum
#SBATCH --partition=standard
#SBATCH -N 1     # nodes
#SBATCH -c 37     # cpus-per-task ("cores")
#SBATCH --output=slurm-output/training-%A.log
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
# target
time python train.py 37 $1
