#!/bin/bash -l
#SBATCH --job-name=float_data_job
#SBATCH --account=NIOW0001
#SBATCH -n 10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH -C skylake
#SBATCH --output=float_data_job.out.%j

source ~/.bashrc
conda activate analysis
### Run program
which python
srun /glade/u/home/mortimer/anaconda3/envs/analysis/bin/python -m process_float.py
