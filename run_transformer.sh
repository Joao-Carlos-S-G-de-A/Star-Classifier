#!/bin/bash
#SBATCH --nodes=1            # Specify number of nodes
#SBATCH --mem=32000            # Specify memory required
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --job-name=Transformer_LAMOST
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=jsga1n21@soton.ac.uk

sbatch Transformer_LAMOST_for_iridis.py
