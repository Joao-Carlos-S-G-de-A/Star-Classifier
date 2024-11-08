#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --ntasks=2
#SBATCH --time=6:00:00
#SBATCH --job-name=CrossViT_Lamost_rope
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jsga1n21@soton.ac.uk

module load python/3.11

# Install necessary packages
pip install --user torch scikit-learn numpy pandas timm
pip install --no-index wandb 

# Uncomment the line below and comment out "wandb offline" if running on Cedar
wandb login 5ffa81406d14f2e8c23a8ec1a882e0b245008ff8
wandb offline

# Confirming loaded modules in the job log
echo "Loaded Modules:"
module list

# Run your script
srun python CrossVit_Lamost_rope.py
