#!/bin/bash
#SBATCH --job-name=test_xgboost_member0
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_xgboost_member0_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_xgboost_member0_%A_%a.err
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

# Experiment to test based on array index
EXPERIMENTS=("exp_1_exp_100" "exp_4_exp_400" "exp_5_exp_500" "exp_6_exp_600" "exp_7_exp_700" "exp_8_exp_800" "exp_9_exp_900")
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

# Activate tf-env (assuming it includes xgboost)
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

# Verify numpy and xgboost
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)"

# Run the test script with the specific lead time and experiment
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/results_xgboost_v050.py --lead $LEAD --experiment "$EXPERIMENT"
echo "Python test script executed for experiment $EXPERIMENT"