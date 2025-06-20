#!/bin/bash
#SBATCH --job-name=test_and_analyze_models
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_and_analyze_models_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/test_and_analyze_models_%A_%a.err
#SBATCH --time=06:00:00
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

experiments=(
    'exp_2/exp_201'
    'exp_3/exp_301'
    'exp_4/exp_401'
    'exp_5/exp_501'
    'exp_6/exp_601'
    'exp_7/exp_701'
    'exp_8/exp_801'
)

EXPERIMENT=${experiments[$SLURM_ARRAY_TASK_ID]}
if [ -z "$EXPERIMENT" ]; then
    echo "Error: EXPERIMENT is not set for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
    exit 1
fi
echo "Testing and analyzing experiment: $EXPERIMENT"

mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2/logs
echo "Log directory created"

source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
echo "Conda initialized"

conda activate tf-env
echo "Conda environment activated: $(/home/tleneman/.conda/envs/tf-env/bin/python --version)"

if ! nvidia-smi; then
    echo "Error: No GPU detected on node $HOSTNAME"
    exit 1
fi

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Verify dependencies
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Available GPUs:', tf.config.list_physical_devices('GPU'))"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import pandas; print('Pandas version:', pandas.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import seaborn; print('Seaborn version:', seaborn.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

# Run the script with experiment argument
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/RMSE_model2.py --experiment "$EXPERIMENT"
echo "Python test and analysis script executed for experiment $EXPERIMENT with RMSE computation"

echo "Job completed at $(date)"