import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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

# Integrated Gradients function
def integrated_gradients(model, input_data, baseline=None, num_steps=50, class_index=1):
    if baseline is None:
        baseline = np.zeros_like(input_data)
    interpolated_inputs = [
        baseline + (float(i) / num_steps) * (input_data - baseline)
        for i in range(num_steps + 1)
    ]
    interpolated_inputs = tf.convert_to_tensor(interpolated_inputs, dtype=tf.float32)
    grads = []
    for inp in interpolated_inputs:
        with tf.GradientTape() as tape:
            tape.watch(inp)
            predictions = model(inp)
            output = predictions[:, class_index]
        grad = tape.gradient(output, inp)
        grads.append(grad)
    avg_grads = tf.reduce_mean(grads, axis=0)
    integrated_grads = (input_data - baseline) * avg_grads
    return integrated_grads.numpy()

# Function to visualize average attributions
def visualize_avg_attributions(pos_attributions, neg_attributions, lead, experiment, output_dir, height=24, width=144, lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0)):
    # Prepare data
    slp_attrs_pos = pos_attributions[0, :3456].reshape(24, 144) if pos_attributions is not None else np.zeros((24, 144))
    v050_attrs_pos = pos_attributions[0, 3456:].reshape(24, 144) if pos_attributions is not None else np.zeros((24, 144))
    slp_attrs_neg = neg_attributions[0, :3456].reshape(24, 144) if neg_attributions is not None else np.zeros((24, 144))
    v050_attrs_neg = neg_attributions[0, 3456:].reshape(24, 144) if neg_attributions is not None else np.zeros((24, 144))
    
    shift_indices = 72
    slp_attrs_pos = np.roll(slp_attrs_pos, shift_indices, axis=1)
    v050_attrs_pos = np.roll(v050_attrs_pos, shift_indices, axis=1)
    slp_attrs_neg = np.roll(slp_attrs_neg, shift_indices, axis=1)
    v050_attrs_neg = np.roll(v050_attrs_neg, shift_indices, axis=1)
    
    # Normalize
    slp_attrs_pos = slp_attrs_pos / np.max(np.abs(slp_attrs_pos)) if np.max(np.abs(slp_attrs_pos)) != 0 else slp_attrs_pos
    v050_attrs_pos = v050_attrs_pos / np.max(np.abs(v050_attrs_pos)) if np.max(np.abs(v050_attrs_pos)) != 0 else v050_attrs_pos
    slp_attrs_neg = slp_attrs_neg / np.max(np.abs(slp_attrs_neg)) if np.max(np.abs(slp_attrs_neg)) != 0 else slp_attrs_neg
    v050_attrs_neg = v050_attrs_neg / np.max(np.abs(v050_attrs_neg)) if np.max(np.abs(v050_attrs_neg)) != 0 else v050_attrs_neg
    
    lons = np.linspace(lon_range[0], lon_range[1], width)
    lats = np.linspace(lat_range[0], lat_range[1], height)
    projection = ccrs.PlateCarree()
    
    # Plot for SLP
    fig_slp = plt.figure(figsize=(14, 4))
    
    ax_slp_pos = fig_slp.add_subplot(121, projection=projection)
    ax_slp_pos.coastlines()
    ax_slp_pos.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh_slp_pos = ax_slp_pos.pcolormesh(lons, lats, np.abs(slp_attrs_pos), transform=ccrs.PlateCarree(), cmap='Reds')
    fig_slp.colorbar(mesh_slp_pos, ax=ax_slp_pos, label='Attribution')
    ax_slp_pos.set_title('SLP Positive High-Confidence')
    ax_slp_pos.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    ax_slp_neg = fig_slp.add_subplot(122, projection=projection)
    ax_slp_neg.coastlines()
    ax_slp_neg.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh_slp_neg = ax_slp_neg.pcolormesh(lons, lats, np.abs(slp_attrs_neg), transform=ccrs.PlateCarree(), cmap='Reds')
    fig_slp.colorbar(mesh_slp_neg, ax=ax_slp_neg, label='Attribution')
    ax_slp_neg.set_title('SLP Negative High-Confidence')
    ax_slp_neg.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    plt.suptitle(f'SLP Average Attributions (Lead {lead})')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    experiment = experiment.replace('/', '_')
    output_path_slp = os.path.join(output_dir, f'slp_avg_ig_lead_{lead}_{experiment.replace(".h5", "")}.png')
    plt.savefig(output_path_slp, bbox_inches='tight')
    plt.close()
    print(f"Saved SLP average IG plot to {output_path_slp}")
    
    # Plot for V050
    fig_v050 = plt.figure(figsize=(14, 4))
    
    ax_v050_pos = fig_v050.add_subplot(121, projection=projection)
    ax_v050_pos.coastlines()
    ax_v050_pos.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh_v050_pos = ax_v050_pos.pcolormesh(lons, lats, np.abs(v050_attrs_pos), transform=ccrs.PlateCarree(), cmap='Reds')
    fig_v050.colorbar(mesh_v050_pos, ax=ax_v050_pos, label='Attribution')
    ax_v050_pos.set_title('V050 Positive High-Confidence')
    ax_v050_pos.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    ax_v050_neg = fig_v050.add_subplot(122, projection=projection)
    ax_v050_neg.coastlines()
    ax_v050_neg.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh_v050_neg = ax_v050_neg.pcolormesh(lons, lats, np.abs(v050_attrs_neg), transform=ccrs.PlateCarree(), cmap='Reds')
    fig_v050.colorbar(mesh_v050_neg, ax=ax_v050_neg, label='Attribution')
    ax_v050_neg.set_title('V050 Negative High-Confidence')
    ax_v050_neg.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    plt.suptitle(f'V050 Average Attributions (Lead {lead})')
    plt.tight_layout()
    output_path_v050 = os.path.join(output_dir, f'v050_avg_ig_lead_{lead}_{experiment.replace(".h5", "")}.png')
    plt.savefig(output_path_v050, bbox_inches='tight')
    plt.close()
    print(f"Saved V050 average IG plot to {output_path_v050}")

# Parse arguments
parser = argparse.ArgumentParser(description="Plot average Integrated Gradients for specific lead and experiment.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories and parameters
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined"
ddir_out = "/gpfs/home4/tleneman/model1_v050/"
analysis_output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/"
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']

# Custom objects
custom_objects = {'PredictionAccuracy': tf.keras.metrics.CategoricalAccuracy}

# Function to load and predict
def predict_with_model(model_path, X_test, Y_test):
    model = load_model(model_path, custom_objects=custom_objects)
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    return Y_pred_prob, Y_pred, Y_true, model

# Load test data
lead = args.lead
experiment = args.experiment
X_test_list = []
Y_test_dict = {}
for ens in test_ens:
    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    if not os.path.exists(X_file):
        print(f"Test file {X_file} not found, exiting")
        exit(1)
    X = load_data(X_file)
    X_test_list.append(X)
    labels_file = os.path.join(ddir_data, f'labels_{ens}.npz')
    if not os.path.exists(labels_file):
        print(f"Labels file {labels_file} not found, exiting")
        exit(1)
    labels_dict = np.load(labels_file)
    lead_key = f'lead_{lead}'
    if lead_key not in labels_dict:
        print(f"Key {lead_key} not found in {labels_file}, exiting")
        exit(1)
    Y_test_dict[lead] = labels_dict[lead_key]

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]

# Initialize storage for IG
ig_correct_positive_high_conf = []
ig_correct_negative_high_conf = []

# Main loop
for seed in seeds:
    model_file = os.path.join(ddir_out, f"model_lead_{lead}_seed_{seed}_{experiment.replace('/', '_')}")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found, skipping...")
        continue
    Y_pred_prob, Y_pred, Y_true, model = predict_with_model(model_file, X_test, Y_test)
    baseline = np.mean(X_test, axis=0, keepdims=True)
    
    # Correct positive predictions
    correct_positive_mask = (Y_pred == 1) & (Y_true == 1) & (Y_pred_prob[:, 1] > 0.622)
    correct_positive_indices = np.where(correct_positive_mask)[0]
    for idx in correct_positive_indices:
        input_data_idx = X_test[idx:idx+1]
        predicted_class_idx = np.argmax(Y_pred_prob[idx])
        ig_attrs_idx = integrated_gradients(model, input_data_idx, baseline=baseline, class_index=predicted_class_idx)
        ig_correct_positive_high_conf.append(ig_attrs_idx)

    # Correct negative predictions
    correct_negative_mask = (Y_pred == 0) & (Y_true == 0) & (Y_pred_prob[:, 0] > 0.622)
    correct_negative_indices = np.where(correct_negative_mask)[0]
    for idx in correct_negative_indices:
        input_data_idx = X_test[idx:idx+1]
        predicted_class_idx = np.argmax(Y_pred_prob[idx])
        ig_attrs_idx = integrated_gradients(model, input_data_idx, baseline=baseline, class_index=predicted_class_idx)
        ig_correct_negative_high_conf.append(ig_attrs_idx)

# Visualize average IG
avg_ig_positive = np.mean(np.concatenate(ig_correct_positive_high_conf, axis=0), axis=0).reshape(1, 6912) if ig_correct_positive_high_conf else None
avg_ig_negative = np.mean(np.concatenate(ig_correct_negative_high_conf, axis=0), axis=0).reshape(1, 6912) if ig_correct_negative_high_conf else None

if avg_ig_positive is not None or avg_ig_negative is not None:
    visualize_avg_attributions(
        avg_ig_positive,
        avg_ig_negative,
        lead, experiment, analysis_output_dir,
        height=24, width=144
    )
else:
    print("No correct predictions with confidence > 0.768 found for either positive or negative classes.")