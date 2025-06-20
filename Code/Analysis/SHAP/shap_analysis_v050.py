import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pandas as pd
import shap
import matplotlib.pyplot as plt
import argparse

# Function to load data
def load_data(file_path, subset_size=None):
    print(f"Attempting to load data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    print(f"Loaded data shape: {data.shape}, dtype: {data.dtype}")
    if subset_size is not None:
        data = data[:subset_size]
        print(f"Subset data to first {subset_size} samples, new shape: {data.shape}")
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    inf_count = np.isinf(data).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {nan_count} NaNs and {inf_count} infinite values detected in {file_path}!")
        if data.ndim == 2:
            column_means = np.nanmean(data, axis=0)
            data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
            print(f"  Replaced NaNs/infs with column means")
        else:
            raise ValueError("NaNs or infinite values found in labels. Please fix preprocessing.")
    return data

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run SHAP analysis for a specific lead time and experiment.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
parser.add_argument('--use-cpu', action='store_true', help="Force SHAP computation on CPU instead of GPU")
args = parser.parse_args()
print(f"Received arguments: lead={args.lead}, experiment={args.experiment}, use_cpu={args.use_cpu}")

# Configure TensorFlow to use CPU if requested
if args.use_cpu:
    tf.config.set_visible_devices([], 'GPU')
    print("Forced TensorFlow to use CPU only")
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"Using GPU: {physical_devices[0]}")
    else:
        print("No GPU available, falling back to CPU")

# Directories
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined/"  # Combined PSL + v050 data
ddir_out = "/gpfs/home4/tleneman/model1_v050/"
analysis_output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/"
shap_output_dir = os.path.join(analysis_output_dir, "shap")
os.makedirs(shap_output_dir, exist_ok=True)
print(f"Output directory set to: {shap_output_dir}, permissions: {os.access(shap_output_dir, os.W_OK)}")

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']
print(f"Using seeds: {seeds}, test ensemble: {test_ens}")

# Define custom objects
custom_objects = {'PredictionAccuracy': tf.keras.metrics.CategoricalAccuracy}

# Main testing loop for the specific lead time and experiment
lead = args.lead
experiment = args.experiment
print(f"Processing lead {lead} with experiment {experiment}")

X_test_list = []
Y_test_dict = {}
for ens in test_ens:
    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    print(f"Loading X data from {X_file}")
    X = load_data(X_file)
    X_test_list.append(X)
    print(f"Appended X data for ensemble {ens}, shape: {X.shape}, total datasets: {len(X_test_list)}")
    
    labels_file = os.path.join("/gpfs/home4/tleneman/Data/Processed_cesm2/test_set_v050/", f'labels_{ens}.npz')  # Corrected path
    print(f"Loading labels from {labels_file}")
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
    print(f"Loaded Y data for lead {lead}, shape: {Y.shape}, dtype: {Y.dtype}")

if not X_test_list:
    print(f"No test data loaded for lead {lead}, skipping")
    exit(1)

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]
print(f"Concatenated X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

# Note the number of features
n_features = X_test.shape[1]
print(f"X_test has {n_features} features, combined PSL and v050 over North Atlantic grid")

# Determine number of classes
n_classes = len(np.unique(Y_test))
print(f"Number of unique classes in Y_test: {n_classes}")
if n_classes < 2:
    print(f"Warning: Only {n_classes} unique classes found in Y_test. SHAP requires at least 2 classes.")
    exit(1)

# Estimate split between PSL and v050 (approximate, adjust if known)
n_features_per_var = n_features // 2  # Assuming equal split
feature_names = [f"PSL_{i//n_features_per_var}_{i%n_features_per_var}" for i in range(n_features)]  # Placeholder names
print(f"Feature names set to: {feature_names[:10]}... (showing first 10 of {n_features}, split approx. {n_features_per_var} per variable)")

for seed in seeds:
    model_file = os.path.join(ddir_out, f"model_lead_{lead}_seed_{seed}_{experiment.replace('/', '_')}")
    print(f"Attempting to load model from {model_file}")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found, skipping seed {seed}...")
        continue
    
    print(f"Loading model for lead {lead}, seed {seed}, experiment {experiment}")
    model = load_model(model_file, custom_objects=custom_objects)
    print(f"Model loaded successfully, input shape: {model.input_shape}, output shape: {model.output_shape}")
    
    # Verify model input shape matches data
    expected_features = model.input_shape[1] if model.input_shape[1] is not None else n_features
    if n_features != expected_features:
        print(f"Warning: X_test has {n_features} features, but model expects {expected_features}. This may cause issues.")
    
    # Compute SHAP values
    print(f"Starting SHAP computation for lead {lead}, seed {seed}")
    try:
        # Use a moderate background size to balance memory and accuracy
        background_size = 1000  # Reduced from 24867 to avoid memory overload
        print(f"Using background size: {background_size}")
        background = X_test[np.random.choice(X_test.shape[0], background_size, replace=False)]
        print(f"Background data selected, shape: {background.shape}")
        explainer = shap.DeepExplainer(model, background)
        print(f"SHAP explainer created")
        shap_values = explainer.shap_values(X_test)
        print(f"SHAP values computed, raw shape: {np.array(shap_values).shape if isinstance(shap_values, list) else shap_values.shape}")
        
        # Handle binary or multi-class classification
        if n_classes == 2:  # Binary classification
            shap_values_raw = shap_values[1]  # Raw SHAP values for positive class
            print(f"Debug: shap_values[1] shape: {shap_values_raw.shape}")
            shap_values_class = shap_values_raw  # Use as is, expecting (24867, 6912)
            print(f"Using SHAP values, shape: {shap_values_class.shape}")
            # Ensure the shape is correct before saving
            if shap_values_class.shape != (X_test.shape[0], X_test.shape[1]):
                raise ValueError(f"Expected SHAP values shape {(X_test.shape[0], X_test.shape[1])}, got {shap_values_class.shape}")
        else:  # Multi-class
            shap_values_class = shap_values  # List of SHAP values for each class
            print(f"Using SHAP values for {n_classes} classes (multi-class), shape: {np.array(shap_values_class).shape}")
        
        # Ensure the output directory exists before saving
        os.makedirs(shap_output_dir, exist_ok=True)
        # Construct filename with proper replacement of all '/' in experiment
        safe_experiment = experiment.replace('/', '_')
        shap_output_file = os.path.join(shap_output_dir, f'shap_values_lead_{lead}_seed_{seed}_{safe_experiment}_new.npz')
        print(f"Saving SHAP values to {shap_output_file}")
        # Save the full shap_values_class array
        np.savez(shap_output_file, shap_values=shap_values_class, feature_names=feature_names)
        print(f"SHAP values saved successfully, file size: {os.path.getsize(shap_output_file)} bytes")
        
        # Generate and save SHAP summary plot with shape check
        print(f"Generating SHAP summary plot")
        if shap_values_class.shape[1] != X_test.shape[1]:
            raise ValueError(f"SHAP values shape {shap_values_class.shape} does not match X_test shape {X_test.shape} on features")
        shap.summary_plot(shap_values_class, X_test, feature_names=feature_names, show=False)
        summary_plot_file = os.path.join(shap_output_dir, f'shap_summary_lead_{lead}_seed_{seed}_{safe_experiment}_new.png')
        print(f"Saving summary plot to {summary_plot_file}")
        plt.savefig(summary_plot_file, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved, file size: {os.path.getsize(summary_plot_file)} bytes")
        
        # Generate and save SHAP summary bar plot
        print(f"Generating SHAP summary bar plot")
        shap.summary_plot(shap_values_class, X_test, feature_names=feature_names, plot_type="bar", show=False)
        bar_plot_file = os.path.join(shap_output_dir, f'shap_summary_bar_lead_{lead}_seed_{seed}_{safe_experiment}_new.png')
        print(f"Saving bar plot to {bar_plot_file}")
        plt.savefig(bar_plot_file, bbox_inches='tight')
        plt.close()
        print(f"SHAP summary bar plot saved, file size: {os.path.getsize(bar_plot_file)} bytes")
        
    except Exception as e:
        print(f"Error computing SHAP values for lead {lead}, seed {seed}: {str(e)}")
        continue
print(f"Completed processing for lead {lead}")