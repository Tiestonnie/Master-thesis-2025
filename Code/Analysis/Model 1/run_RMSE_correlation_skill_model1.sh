#!/bin/bash
#SBATCH --job-name=nao_metrics_analysis
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/nao_metrics_analysis_%j.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/nao_metrics_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100
#SBATCH --gpus-per-node=1
#SBATCH --exclude=tcn331
#SBATCH --array=0-6

echo "Starting job at $(date)"
echo "Running on node: $HOSTNAME"

# Create log directory
mkdir -p /gpfs/home4/tleneman/Data/Processed_cesm2/logs
echo "Log directory created"

# Initialize Conda
source /sw/arch/RHEL9/EB_production/2024/software/Miniconda3/24.7.1-0/etc/profile.d/conda.sh
echo "Conda initialized"

# Activate tf-env
conda activate tf-env
echo "Conda environment activated: $(/home/tleneman/.conda/envs/tf-env/bin/python --version)"

# Verify numpy, pandas, matplotlib, seaborn, scikit-learn
/home/tleneman/.conda/envs/tf-env/bin/python -c "import numpy; print('NumPy version:', numpy.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import pandas; print('Pandas version:', pandas.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import matplotlib; print('Matplotlib version:', matplotlib.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import seaborn; print('Seaborn version:', seaborn.__version__)"
/home/tleneman/.conda/envs/tf-env/bin/python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"

# Run the analysis script
/home/tleneman/.conda/envs/tf-env/bin/python /gpfs/home4/tleneman/RMSE_correlation_skill_model1.py
echo "Python analysis script executed"

echo "Job completed at $(date)"