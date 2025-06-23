#!/bin/bash
#SBATCH --job-name=test_model1_member0
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_model1_member0_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_model1_member0_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"

# Experiment to test based on array index
#EXPERIMENTS=("exp_1/exp_100.h5" "exp_4/exp_400.h5" "exp_5/exp_500.h5" "exp_6/exp_600.h5" "exp_7/exp_700.h5" "exp_8/exp_800.h5" "exp_9/exp_900.h5")
EXPERIMENTS=("exp_1/exp_100.h5")
LEAD_TIMES=(14 21 28 35 42 49 56)
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}
LEAD=${LEAD_TIMES[$SLURM_ARRAY_TASK_ID]}
echo "Testing experiment: $EXPERIMENT for lead $LEAD days"

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

# Run the test script with the specific lead time and experiment
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/ig_model1_v050_plot5.py --lead $LEAD --experiment "$EXPERIMENT"
echo "Python test script executed for experiment $EXPERIMENT"