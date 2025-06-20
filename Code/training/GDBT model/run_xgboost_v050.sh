#!/bin/bash
#SBATCH --job-name=xgboost_conda
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2_combined/logs/xgboost_slurm_conda_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2_combined/logs/xgboost_slurm_conda_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0-4

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# List of experiments
experiments=("exp_4/exp_400")

# Get the experiment
EXPERIMENT=${experiments[0]}
if [ -z "$EXPERIMENT" ]; then
    echo "Error: EXPERIMENT is not set for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Running experiment: $EXPERIMENT"

# Create log directory
mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2_combined/logs
echo "Log directory created"

# Initialize Conda
source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
echo "Conda initialized"

# Activate tf-env
conda activate tf-env
echo "Conda environment activated: $(/home/tleneman/.conda/envs/tf-env/bin/python --version)"

# Check GPU availability
if ! nvidia-smi; then
    echo "Error: No GPU detected on node $HOSTNAME"
    exit 1
fi

# Set LD_LIBRARY_PATH for CUDA (optional for XGBoost, but kept for consistency)
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify numpy and xgboost
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"

# Define seed subsets
seed_subsets=([0]="210 47" [1]="33 133" [2]="410 692" [3]="64 99" [4]="910 92")
RANDOM_SEED="${seed_subsets[$SLURM_ARRAY_TASK_ID]}"
echo "Running with seeds: $RANDOM_SEED"

# Run the XGBoost script
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/xgboost_v050_slurm.py "$EXPERIMENT" "$RANDOM_SEED"
echo "Python script executed for experiment $EXPERIMENT"