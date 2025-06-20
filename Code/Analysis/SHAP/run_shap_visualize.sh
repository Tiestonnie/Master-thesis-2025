#!/bin/bash
#SBATCH --job-name=visualize_shap
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/visualize_shap_%j.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/visualize_shap_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0







# Activate the Conda environment
source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
conda activate tf-env

# Run the Python script
python /gpfs/home4/tleneman/shap_visualize.py

echo "Script execution completed at $(date)"