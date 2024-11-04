#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=32000
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=2-12:00:00
#SBATCH --job-name=Transformer_LAMOST
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jsga1n21@soton.ac.uk


module load python/3.11
pip install --user torch scikit-learn numpy pandas
pip install --no-index wandb
# Check if other necessary modules are available, or else use Python's pip installation
module load torch  # Check if PyTorch is available as a module
module load numpy  # Check if NumPy is available as a module
module load pandas  # Check if Pandas is available as a module
module load sklearn  # Check if scikit-learn is available as a module
module load wandb  # Load wandb if available



### Save your wandb API key in your .bash_profile or replace $API_KEY with your actual API key. Uncomment the line below and comment out "wandb offline" if running on Cedar ###

wandb login 5ffa81406d14f2e8c23a8ec1a882e0b245008ff8

wandb offline


# Confirming loaded modules in the job log
echo "Loaded Modules:"
module list

srun python Transformer_LAMOST_for_iridis_trial.py

