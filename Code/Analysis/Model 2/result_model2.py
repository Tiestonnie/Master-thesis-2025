import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import sys

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

# --- Testing Phase ---

# Directories
#ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2/test_set_3days/"
ddir_data = "/gpfs/home4/tleneman/Data/Processed_era5/"
ddir_out = "/gpfs/home4/tleneman/model2_3day/"
#output_dir = "/gpfs/home4/tleneman/model2_3day/results_test_set/"
output_dir = "/gpfs/home4/tleneman/model2_3day/results_test_set_era5/"
os.makedirs(output_dir, exist_ok=True)

# Parameters
experiments = [
    'exp_2/exp_201',
    'exp_3/exp_301',
    'exp_4/exp_401',
    'exp_5/exp_501',
    'exp_6/exp_601',
    'exp_7/exp_701',
    'exp_8/exp_801',
]
lead_times = [14, 21, 28, 35, 42, 49, 56]
seeds = [19, 26, 54, 68, 6]
test_ens = ['0']

# Define custom objects
custom_objects = {'PredictionAccuracy': tf.keras.metrics.CategoricalAccuracy}

# Get the experiment from the environment variable
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
if task_id >= len(experiments):
    print(f"Error: SLURM_ARRAY_TASK_ID {task_id} exceeds number of experiments.")
    sys.exit(1)
experiment = experiments[task_id]
lead = lead_times[task_id]

# Load test data for this lead time
X_test_list = []
Y_test_dict = {}
for ens in test_ens:
#    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    X_file = os.path.join(ddir_data, f'3day_member_{ens}_lead_{lead}.npy')
    if not os.path.exists(X_file):
        print(f"Test file {X_file} not found, skipping lead {lead}")
        sys.exit(1)
    X = load_data(X_file)
    X_test_list.append(X)
    
    labels_file = os.path.join(ddir_data, f'labels_{ens}.npz')
    if not os.path.exists(labels_file):
        print(f"Labels file {labels_file} not found, skipping lead {lead}")
        sys.exit(1)
    labels_dict = np.load(labels_file)
    lead_key = f'lead_{lead}'
    if lead_key not in labels_dict:
        print(f"Key {lead_key} not found in {labels_file}, skipping lead {lead}")
        sys.exit(1)
    Y = labels_dict[lead_key]
    Y_test_dict[lead] = Y

if not X_test_list:
    print(f"No test data loaded for lead {lead}, skipping")
    sys.exit(1)

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]
print(f"Test set for lead {lead} days: X_test shape {X_test.shape}, Y_test shape {Y_test.shape}")
print(f"Y_test distribution (class 0 %): {np.mean(Y_test[:, 0]):.4f}")

# Function to load and predict with a model
def predict_with_model(model_path, X_test, Y_test, lead, seed, experiment):
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Model input shape expected: {model.input_shape}, X_test shape: {X_test.shape}")
        print(f"Model output shape expected: {model.output_shape}, Y_test shape: {Y_test.shape}")
        Y_pred_prob = model.predict(X_test, verbose=0)
        
        # Handle model output shape
        if Y_pred_prob.shape[1] == 1:
            print("Warning: Model outputs a single value. Assuming binary classification with sigmoid activation.")
            Y_pred_prob = np.hstack([1 - Y_pred_prob, Y_pred_prob])  # Convert to two-class probabilities
        elif Y_pred_prob.shape[1] != 2:
            raise ValueError(f"Unexpected model output shape {Y_pred_prob.shape}. Expected (None, 2) or (None, 1).")
        
        Y_pred = np.argmax(Y_pred_prob, axis=1)
        Y_true = np.argmax(Y_test, axis=1)
        accuracy = accuracy_score(Y_true, Y_pred)
        print(f"Lead {lead} days, Seed {seed}, Experiment {experiment} - Test Accuracy: {accuracy:.4f}")
        return Y_pred_prob, Y_pred, Y_true, accuracy
    except Exception as e:
        print(f"Error loading or predicting with model {model_path}: {str(e)}")
        return None, None, None, 0.0

# Test the model
for seed in seeds:
    model_file = os.path.join(ddir_out, f"model_3day_lead_{lead}_seed_{seed}_{experiment.replace('/', '_')}.h5")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found, skipping... Please verify the model file exists in {ddir_out}")
        continue
    
    print(f"\nTesting model for lead {lead} days, seed {seed}, experiment {experiment}")
    Y_pred_prob, Y_pred, Y_true, accuracy = predict_with_model(model_file, X_test, Y_test, lead, seed, experiment)
    
    if Y_pred_prob is not None:
        output_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.npz')
        np.savez(
            output_file,
            confidences=Y_pred_prob,
            predictions=Y_pred,
            true_labels=Y_true
        )
        print(f"Saved predictions and confidences to {output_file}")

# --- Analysis and Visualization Phase ---

analysis_output_dir = "/gpfs/home4/tleneman/model2_3day/results_test_set/analysis/"
os.makedirs(analysis_output_dir, exist_ok=True)

# Use lead-specific CSV filename
accuracy_csv = os.path.join(analysis_output_dir, f'accuracy_by_confidence_bins_lead_{lead}.csv')
accuracy_df = pd.DataFrame(columns=['lead', 'experiment', 'seed', 'bin_1_accuracy', 'bin_2_accuracy', 'bin_3_accuracy', 
                                    'bin_4_accuracy', 'bin_5_accuracy', 'bin_6_accuracy', 'bin_7_accuracy', 
                                    'bin_8_accuracy', 'bin_9_accuracy', 'bin_10_accuracy', 'overall_accuracy'])

for seed in seeds:
    npz_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.npz')
    if not os.path.exists(npz_file):
        print(f"Results file {npz_file} not found, skipping... Please verify the output file was created.")
        continue
    
    data = np.load(npz_file)
    confidences = data['confidences']
    predictions = data['predictions']
    true_labels = data['true_labels']
    
    confidence_scores = np.max(confidences, axis=1)
    
    df_dict = {
        'true_label': true_labels,
        'predicted_label': predictions,
        'confidence_score': confidence_scores,
        'confidence_class_0': confidences[:, 0]
    }
    if confidences.shape[1] > 1:
        df_dict['confidence_class_1'] = confidences[:, 1]
    df = pd.DataFrame(df_dict)
    
    df_sorted = df.sort_values(by='confidence_score', ascending=True)
    
    csv_file = os.path.join(analysis_output_dir, f'sorted_predictions_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.csv')
    df_sorted.to_csv(csv_file, index=False)
    print(f"Saved sorted predictions to {csv_file}")
    
    n_samples = len(df)
    bin_size = n_samples // 10
    bins = [df_sorted.iloc[i * bin_size:(i + 1) * bin_size] for i in range(10)]
    if n_samples % 10 != 0:
        bins[-1] = df_sorted.iloc[9 * bin_size:]
    
    bin_accuracies = []
    for i, bin_df in enumerate(bins):
        bin_accuracy = accuracy_score(bin_df['true_label'], bin_df['predicted_label'])
        bin_accuracies.append(bin_accuracy)
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Bin {i+1} (Confidence {bin_df['confidence_score'].min():.4f} to {bin_df['confidence_score'].max():.4f}): Accuracy = {bin_accuracy:.4f}")
    
    overall_accuracy = accuracy_score(true_labels, predictions)
    print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Overall Accuracy: {overall_accuracy:.4f}")
    
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame([{
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
        'overall_accuracy': overall_accuracy
    }])], ignore_index=True)

accuracy_df.to_csv(accuracy_csv, index=False)
print(f"Saved accuracy by confidence bins to {accuracy_csv}")