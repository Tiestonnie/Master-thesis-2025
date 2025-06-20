#!/bin/bash
#SBATCH --job-name=model1_conda
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/model1_slurm_conda_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/model1_slurm_conda_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0-5

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# List of experiments
experiments=("exp_1/exp_104" "exp_1/exp_105" "exp_1/exp_106" "exp_1/exp_107" "exp_1/exp_108" "exp_1/exp_109")

# Get the experiment
EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}
if [ -z "$EXPERIMENT" ]; then
    echo "Error: EXPERIMENT is not set for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Running experiment: $EXPERIMENT"

# Create log directory
mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2/logs
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

# Run the script
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/model1_prototype_slurm.py "$EXPERIMENT"
echo "Python script executed for experiment $EXPERIMENT"