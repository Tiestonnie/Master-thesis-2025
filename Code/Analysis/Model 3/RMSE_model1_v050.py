import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, mean_squared_error
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Function to load data
def load_data(file_path, subset_size=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    if subset_size is not None:
        data = data[:subset_size]
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    inf_count = np.isinf(data).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {nan_count} NaNs and {inf_count} infinite values detected in {file_path}!")
        if data.ndim == 2:
            column_means = np.nanmean(data, axis=0)
            data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
        else:
            raise ValueError("NaNs or infinite values found in labels. Please fix preprocessing.")
    return data

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run model testing for a specific lead time and experiment.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories ERA 5
ddir_data = "/gpfs/home4/tleneman/Data/Processed_era5/v10/"
ddir_out = "/gpfs/home4/tleneman/model1_v050/"
output_dir = "/gpfs/home4/tleneman/model1_v050/results_v10_set/"
os.makedirs(output_dir, exist_ok=True)
analysis_output_dir = "/gpfs/home4/tleneman/model1_v050/results_v10_set/analysis/"
os.makedirs(analysis_output_dir, exist_ok=True)



# Directories CESM2
#ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2/test_set_v050/"
#ddir_out = "/gpfs/home4/tleneman/model1_v050/"
#output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_set/"
#os.makedirs(output_dir, exist_ok=True)
#analysis_output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/"
#os.makedirs(analysis_output_dir, exist_ok=True)

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']
experiments = [
    'exp_1/exp_100.h5',
    'exp_4/exp_400.h5',
    'exp_5/exp_500.h5',
    'exp_6/exp_600.h5',
    'exp_7/exp_700.h5',
    'exp_8/exp_800.h5',
    'exp_9/exp_900.h5'
]
lead_times = [14, 21, 28, 35, 42, 49, 56]

# Define custom objects
custom_objects = {'PredictionAccuracy': tf.keras.metrics.CategoricalAccuracy}

# Function to load and predict with a model
def predict_with_model(model_path, X_test, Y_test, lead, seed, experiment):
    model = load_model(model_path, custom_objects=custom_objects)
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_true, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_true.astype(float), Y_pred_prob[:, 1]))  # RMSE for positive phase
    print(f"Lead {lead} days, Seed {seed}, Experiment {experiment} - Test Accuracy: {accuracy:.4f}, RMSE: {rmse:.4f}")
    return Y_pred_prob, Y_pred, Y_true, accuracy, rmse

# Main testing loop for the specific lead time and experiment
lead = args.lead
experiment = args.experiment
results = {'confidences': [], 'predictions': [], 'true_labels': [], 'accuracies': [], 'seeds': [], 'rmse': []}

X_test_list = []
Y_test_dict = {}
for ens in test_ens:
    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    if not os.path.exists(X_file):
        print(f"Test file {X_file} not found, skipping lead {lead}")
        exit(1)
    X = load_data(X_file)
    X_test_list.append(X)
    
    labels_file = os.path.join(ddir_data, f'labels_{ens}.npz')
    if not os.path.exists(labels_file):
        print(f"Labels file {labels_file} not found, skipping lead {lead}")
        exit(1)
    labels_dict = np.load(labels_file)
    lead_key = f'lead_{lead}'
    if lead_key not in labels_dict:
        print(f"Key {lead_key} not found in {labels_file}, skipping lead {lead}")
        exit(1)
    Y = labels_dict[lead_key]
    Y_test_dict[lead] = Y

if not X_test_list:
    print(f"No test data loaded for lead {lead}, skipping")
    exit(1)

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]
print(f"Test set for lead {lead} days: X_test shape {X_test.shape}, Y_test shape {Y_test.shape}")

for seed in seeds:
    model_file = os.path.join(ddir_out, f"model_lead_{lead}_seed_{seed}_{experiment.replace('/', '_')}")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found, skipping...")
        continue
    
    print(f"\nTesting model for lead {lead} days, seed {seed}, experiment {experiment}")
    Y_pred_prob, Y_pred, Y_true, accuracy, rmse = predict_with_model(model_file, X_test, Y_test, lead, seed, experiment)
    
    results['confidences'].append(Y_pred_prob)
    results['predictions'].append(Y_pred)
    results['true_labels'].append(Y_true)
    results['accuracies'].append(accuracy)
    results['seeds'].append(seed)
    results['rmse'].append(rmse)
    
    output_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz')
    np.savez(
        output_file,
        confidences=Y_pred_prob,
        predictions=Y_pred,
        true_labels=Y_true
    )
    print(f"Saved predictions and confidences to {output_file}")

# --- Analysis and Visualization Phase ---
accuracy_df = pd.DataFrame(columns=['lead', 'experiment', 'seed', 'bin_1_accuracy', 'bin_2_accuracy', 'bin_3_accuracy', 
                                    'bin_4_accuracy', 'bin_5_accuracy', 'bin_6_accuracy', 'bin_7_accuracy', 
                                    'bin_8_accuracy', 'bin_9_accuracy', 'bin_10_accuracy', 'overall_accuracy', 'rmse'])

print(f"Starting analysis for lead {lead}, experiment {experiment}, seeds {results['seeds']}")

for seed_idx, seed in enumerate(results['seeds']):
    npz_file = os.path.join(output_dir, f"predictions_lead_{lead}_seed_{seed}_{experiment.replace('.h5', '')}_new.npz")
    print(f"Checking npz_file: {npz_file}, exists: {os.path.exists(npz_file)}")
    if not os.path.exists(npz_file):
        print(f"Results file {npz_file} not found, skipping seed {seed}...")
        continue
    
    try:
        data = np.load(npz_file)
        confidences = data['confidences']
        predictions = data['predictions']
        true_labels = data['true_labels']
        
        confidence_scores = np.max(confidences, axis=1)
        df = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence_score': confidence_scores,
            'confidence_class_0': confidences[:, 0],
            'confidence_class_1': confidences[:, 1]
        })
        
        df_sorted = df.sort_values(by='confidence_score', ascending=True)
        # Corrected directory for sorted predictions
        csv_file = os.path.join(analysis_output_dir, f"sorted_predictions_lead_{lead}_seed_{seed}_{experiment.replace('.h5', '')}_new.csv")
        # Create directory for CSV file
        csv_dir = os.path.dirname(csv_file)
        os.makedirs(csv_dir, exist_ok=True)
        df_sorted.to_csv(csv_file, index=False)
        print(f"Saved sorted predictions to {csv_file}")
        
        n_samples = len(df)
        bin_size = n_samples // 10
        bins = [df_sorted.iloc[i * bin_size:(i + 1) * bin_size] for i in range(10)]
        if n_samples % 10 != 0:
            bins[-1] = df_sorted.iloc[9 * bin_size:]
        
        bin_accuracies = []
        for i, bin_df in enumerate(bins):
            if not bin_df.empty:
                bin_accuracy = accuracy_score(bin_df['true_label'], bin_df['predicted_label'])
            else:
                bin_accuracy = 0.0
            bin_accuracies.append(bin_accuracy)
            print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Bin {i+1} Accuracy = {bin_accuracy:.4f}")
        
        overall_accuracy = accuracy_score(true_labels, predictions)
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Overall Accuracy: {overall_accuracy:.4f}")
        
        row_data = {
            'lead': lead,
            'experiment': experiment,
            'seed': seed,
            'bin_1_accuracy': bin_accuracies[0],
            'bin_2_accuracy': bin_accuracies[1],
            'bin_3_accuracy': bin_accuracies[2],
            'bin_4_accuracy': bin_accuracies[3],
            'bin_5_accuracy': bin_accuracies[4],
            'bin_6_accuracy': bin_accuracies[5],
            'bin_7_accuracy': bin_accuracies[6],
            'bin_8_accuracy': bin_accuracies[7],
            'bin_9_accuracy': bin_accuracies[8],
            'bin_10_accuracy': bin_accuracies[9],
            'overall_accuracy': overall_accuracy,
            'rmse': results['rmse'][seed_idx]
        }
        row_df = pd.DataFrame([row_data])
        accuracy_df = pd.concat([accuracy_df, row_df], ignore_index=True)
        print(f"Added row to accuracy_df for seed {seed}: {row_data}")
    except Exception as e:
        print(f"Error processing seed {seed}, npz_file {npz_file}: {e}")
        continue

# Write RMSE CSV with robust error handling
rmse_csv = os.path.join(analysis_output_dir, 'rmse_all_leads.csv')
print(f"Preparing to write CSV: {rmse_csv}")
try:
    # Verify directory is writable
    if not os.access(analysis_output_dir, os.W_OK):
        raise PermissionError(f"No write permission for {analysis_output_dir}")
    # Create analysis_output_dir if it doesn't exist
    os.makedirs(analysis_output_dir, exist_ok=True)
    # Check if accuracy_df is empty
    if accuracy_df.empty:
        print(f"Warning: accuracy_df is empty for lead {lead}, experiment {experiment}. Creating CSV with headers only.")
        accuracy_df.to_csv(rmse_csv, index=False)
        print(f"Created {rmse_csv} with headers only (no data).")
    else:
        # Sort by lead time
        accuracy_df = accuracy_df.sort_values(by='lead')
        # If CSV exists, append; otherwise, create new
        if os.path.exists(rmse_csv):
            print(f"Appending to existing {rmse_csv}")
            existing_df = pd.read_csv(rmse_csv)
            accuracy_df = pd.concat([existing_df, accuracy_df], ignore_index=True)
            accuracy_df = accuracy_df.sort_values(by='lead')
        accuracy_df.to_csv(rmse_csv, index=False)
        print(f"Created/Updated {rmse_csv} with {len(accuracy_df)} rows")
except Exception as e:
    print(f"Error writing {rmse_csv}: {e}")
    raise

# Generate consolidated RMSE plot (only for the last experiment)
if experiment == experiments[-1]:
    try:
        plt.figure(figsize=(10, 6))
        rmse_df = pd.read_csv(rmse_csv)
        rmse_df = rmse_df.sort_values(by='lead')
        for seed in seeds:
            seed_df = rmse_df[rmse_df['seed'] == seed]
            if not seed_df.empty:
                plt.plot(seed_df['lead'], seed_df['rmse'], marker='o', label=f'Seed {seed}')
        plt.xlabel('Lead Time (days)')
        plt.ylabel('RMSE')
        plt.title('RMSE by Lead Time Across All Experiments')
        plt.legend()
        plt.grid(True)
        plot_file = os.path.join(analysis_output_dir, 'rmse_all_leads.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved consolidated RMSE plot to {plot_file}")
    except Exception as e:
        print(f"Error generating RMSE plot: {e}")