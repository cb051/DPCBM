#!/bin/bash


#SBATCH --partition=agpu06
#SBATCH --time=06:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH -c64
#SBATCH --gres=gpu

##SBATCH --output=slurm_outputs/%j.out      ### optional Slurm Output file, %x is job name, %j is job id
##SBATCH --error=slurm_outputs/%j.err       ### optional Slurm Error file, %x is job name, %j is job id

module load python/anaconda-3.14
source /share/apps/python/anaconda-3.14/etc/profile.d/conda.sh 
conda activate CEM_env

python3 models/cbm.py
python3 models/cem.py

##python3 /home/cb051/research/DP_XAI/code/experiments/scbm_cem_implementation.py

##srun -N1 -n1 -c64 -p agpu06 -q gpu -t 06:00:00 --pty $SHELL