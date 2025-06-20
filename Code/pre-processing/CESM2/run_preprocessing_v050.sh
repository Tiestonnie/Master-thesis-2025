#!/bin/bash
#SBATCH --job-name=preprocess_job
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/preprocess_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/preprocess_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=genoa
#SBATCH --array=0-9

# Load modules
#%module purge
#%module load 2022
#%module load Python/3.10.4-GCCcore-11.3.0

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Create log directory
mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2/logs
echo "Log directory created"

# Initialize Conda
source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
echo "Conda initialized"

# Activate tf-env
conda activate tf-env
echo "Conda environment activated: $(/home/tleneman/.conda/envs/tf-env/bin/python --version)"

# Verify required packages
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import xarray; print('xarray version:', xarray.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import scipy; print('SciPy version:', scipy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import pandas; print('Pandas version:', pandas.__version__)"

# Run the preprocessing script
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/v050_preprocessing.py "$SLURM_ARRAY_TASK_ID"
echo "Preprocessing script executed for ensemble $SLURM_ARRAY_TASK_ID"