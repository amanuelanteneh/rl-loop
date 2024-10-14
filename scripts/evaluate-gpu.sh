#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -A quantum
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -c 5   # cpus-per-task ("cores")
#SBATCH --output=evaluate-%A.log
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
# num epsisodes, model name, verify
time python evaluate.py $1 $2 $3
