#!/bin/bash
#
#SBATCH --time=7-01:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,gpub
#SBATCH --mem=10000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.arora@tu-braunschweig.de

module load anaconda/3-5.0.1

source activate deepl_1

srun python -u project_2.py
