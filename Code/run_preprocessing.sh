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


# Activate virtual environment
source /gpfs/home4/tleneman/venvs/preprocess_env/bin/activate

# Verify Python
which python3

# Create logs directory
mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2/logs

# Run preprocessing
python3 /gpfs/home4/tleneman/preprocessing_array.py $SLURM_ARRAY_TASK_ID