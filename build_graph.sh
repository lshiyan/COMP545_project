#!/bin/bash
#SBATCH --output=build_graph.log
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=def-sreddy
module load python/3.10
module load cuda/12.2

# (Optional) activate your virtual environment
source .venv/bin/activate

# Run your script
python3 build_graph.py
