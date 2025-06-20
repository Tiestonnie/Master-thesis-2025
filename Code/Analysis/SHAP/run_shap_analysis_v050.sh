#!/bin/bash
#SBATCH --job-name=shap_v050
#SBATCH --output=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/shap_v050_%A_%a.out
#SBATCH --error=/gpfs/home4/tleneman/Data/Processed_cesm2/logs/shap_v050_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-6

# Load modules and activate Conda
module load 2023
module load Anaconda3/2023.07-2
source $EBROOTANACONDA3/etc/profile.d/conda.sh
conda activate tf-new  # Use new environment if created

# Set lead times
lead_times=(14 21 28 35 42 49 56)
lead=${lead_times[$SLURM_ARRAY_TASK_ID]}

# Run the script
python /gpfs/home4/tleneman/shap_analysis_v050.py --lead $lead --experiment exp_4/exp_400.h5