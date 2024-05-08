#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -A quantum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH -c 100   # cpus-per-task ("cores")
#SBATCH --output=slurm-output/training-%A.log
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
# target
time python train.py 100 $1
