#!/bin/bash
#SBATCH --time=1-00:00:00
#SBATCH -A quantum
#SBATCH --partition=standard
#SBATCH -N 1     # nodes
#SBATCH -c 1     # cpus-per-task ("cores")
#SBATCH --output=evaluate-%A.log
#SBATCH --mail-user=asa2rc@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=ALL

module load anaconda
# num epsisodes, model name
time python evaluate.py $1 $2 $3
