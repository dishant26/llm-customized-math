#!/bin/bash
#SBATCH --job-name=train_sec_qa
#SBATCH --output=train_sec_qa_%j.out
#SBATCH --error=train_sec_qa_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=24:00:00

# Load required modules (modify according to your system)
module load cuda/11.8
module load python/3.9

# Activate virtual environment (modify path as needed)
source /path/to/your/venv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Prepare data
python prepare_data.py

# Run training
python train.py 