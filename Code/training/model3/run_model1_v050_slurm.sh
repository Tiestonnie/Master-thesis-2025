#!/bin/bash
#SBATCH --job-name=model1_conda
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2_combined/logs/model1_slurm_conda_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2_combined/logs/model1_slurm_conda_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0-6

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# List of experiments
#experiments=("exp_1/exp_100" "exp_4/exp_400" "exp_5/exp_500" "exp_6/exp_600" "exp_7/exp_700" "exp_8/exp_800" "exp_9/exp_900")
experiments=("exp_1/exp_100")
# Get the experiment
EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}
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

# Set LD_LIBRARY_PATH for CUDA
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify numpy and TensorFlow
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Available GPUs:', tf.config.list_physical_devices('GPU'))"

# Define seed sets for each experiment
seed_sets=([0]="210 47 33 133 410" [1]="210 47 33 133 410" [2]="210 47 33 133 410" [3]="210 47 33 133 410" [4]="210 47 33 133 410" [5]="210 47 33 133 410" [6]="210 47 33 133 410")
RANDOM_SEED="${seed_sets[$SLURM_ARRAY_TASK_ID]}"
echo "Running with seeds: $RANDOM_SEED"

# Run the script
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/model1_v050_slurm.py "$EXPERIMENT" "$RANDOM_SEED"
echo "Python script executed for experiment $EXPERIMENT"