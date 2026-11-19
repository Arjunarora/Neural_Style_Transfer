#!/bin/bash
#
#SBATCH --time=7-01:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu,gpub
#SBATCH --mem=10000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=<your-email>

module load anaconda/3-5.0.1

source activate deepl_1

srun python -u src/neural_style_transfer.py --content images/example_3/content.jpg --style images/example_3/style.jpg --output result.jpg
