import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Function to load data
def load_data(file_path, subset_size=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    print(f' dit is een andere min  and max of last sample in {file_path}: {data[-1].min():.2f}, {data[-1].max():.2f}')
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
        print(f'min  and max of last sample in {file_path}: {data[-1].min():.2f}, {data[-1].max():.2f}')
    return data

# Integrated Gradients functie
def integrated_gradients(model, input_data, baseline=None, num_steps=50, class_index=1):
    """
    Bereken Integrated Gradients voor een enkele input.
    Args:
        model: Het geladen Keras-model.
        input_data: Inputdata met shape [1, 6912].
        baseline: Referentie-input (default: nul-baseline).
        num_steps: Aantal stappen voor numerieke integratie.
        class_index: Index van de klasse (0=negatief NAO, 1=positief NAO).
    Returns:
        ig_attrs: Attributies met shape [1, 6912].
    """
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

# Function to collect and visualize all seed attributions in one plot
def visualize_all_attributions_cartopy(seed_data, lead, experiment, output_dir, height=24, width=144, channels=2, lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0)):
    """
    Visualize Integrated Gradients for all seeds in a single figure.
    Args:
        seed_data: List of tuples (seed, input_data, attributions, sample_idx).
        lead: Lead time in dagen.
        experiment: Experiment naam.
        output_dir: Directory om heatmap op te slaan.
        height: Hoogte van de grid (24 breedtegraden).
        width: Breedte van de grid (144 lengtegraden).
        channels: Aantal kanalen (2 voor SLP en v050).
        lat_range: Tuple van (min_lat, max_lat).
        lon_range: Tuple van (min_lon, max_lon).
    """
    # Create a 5x2 subplot grid (5 seeds, 2 columns: SLP input, SLP attributions)
    fig = plt.figure(figsize=(14, 20))
    projection = ccrs.PlateCarree()
    lons = np.linspace(lon_range[0], lon_range[1], width)
    lats = np.linspace(lat_range[0], lat_range[1], height)
    shift_indices = 72

    for i, (seed, input_data, attributions, sample_idx) in enumerate(seed_data):
        # Extraheer SLP
        slp_input = input_data[0, :3456].reshape(24, 144)
        slp_attrs = attributions[0, :3456].reshape(24, 144)
        
        slp_input = np.roll(slp_input, shift_indices, axis=1)
        slp_attrs = np.roll(slp_attrs, shift_indices, axis=1)
        
        # Normaliseer attributies
        slp_attrs = slp_attrs / np.max(np.abs(slp_attrs)) if np.max(np.abs(slp_attrs)) != 0 else slp_attrs
        
        # Print min/max for verification
        print(f"Seed {seed}, Sample {sample_idx} - SLP kanaal min: {slp_input.min():.2f}, max: {slp_input.max():.2f}")
        
        # Plot SLP input
        ax_input = fig.add_subplot(5, 2, 2*i + 1, projection=projection)
        ax_input.coastlines()
        ax_input.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh_input = ax_input.pcolormesh(lons, lats, slp_input, transform=ccrs.PlateCarree(), cmap='coolwarm')
        fig.colorbar(mesh_input, ax=ax_input, label='SLP (h Dit is een andere hPa)')
        ax_input.set_title(f'SLP Input (Seed {seed}, Sample {sample_idx})')
        ax_input.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
        
        # Plot SLP attributions
        ax_attrs = fig.add_subplot(5, 2, 2*i + 2, projection=projection)
        ax_attrs.coastlines()
        ax_attrs.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh_attrs = ax_attrs.pcolormesh(lons, lats, np.abs(slp_attrs), transform=ccrs.PlateCarree(), cmap='Reds')
        fig.colorbar(mesh_attrs, ax=ax_attrs, label='Attribution')
        ax_attrs.set_title(f'SLP Attributions (Seed {seed}, Sample {sample_idx})')
        ax_attrs.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

    plt.suptitle('IG for correctly positive NAOI predictions within forecasts of opportunity for 14 day lead time')
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    experiment = experiment.replace('/', '_')
    output_path = os.path.join(output_dir, f'ig_heatmap_cartopy_lead_{lead}_{experiment.replace(".h5", "")}_all_seeds.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved combined Cartopy Integrated Gradients heatmap to {output_path}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run model testing for a specific lead time and experiment.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined"
ddir_out = "/gpfs/home4/tleneman/model1_v050/"
output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_set/"
analysis_output_dir = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(analysis_output_dir, exist_ok=True)

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']

# Define custom objects
custom_objects = {'PredictionAccuracy': tf.keras.metrics.CategoricalAccuracy}

# Function to load and predict with a model
def predict_with_model(model_path, X_test, Y_test, lead, seed, experiment):
    model = load_model(model_path, custom_objects=custom_objects)
    Y_pred_prob = model.predict(X_test)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    accuracy = accuracy_score(Y_true, Y_pred)
    print(f"Lead {lead} days, Seed {seed}, Experiment {experiment} - Test Accuracy: {accuracy:.4f}")
    return Y_pred_prob, Y_pred, Y_true, accuracy

# Main testing loop
lead = args.lead
experiment = args.experiment
results = {'confidences': [], 'predictions': [], 'true_labels': [], 'accuracies': [], 'seeds': []}
seed_data = []  # Store data for visualization

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
    Y_pred_prob, Y_pred, Y_true, accuracy = predict_with_model(model_file, X_test, Y_test, lead, seed, experiment)
    
    # Selecteer een willekeurige hoge-confidence sample voor klasse 0 die correct is voorspeld
    confidence_scores = Y_pred_prob[:, 0]
    correct_mask = (Y_pred == Y_true) & (Y_pred == 0)
    high_conf_correct_indices = np.where(correct_mask & (confidence_scores > 0.768))[0]
    if len(high_conf_correct_indices) == 0:
        print(f"No correct high-confidence samples found for class 1, falling back to highest confidence correct sample")
        correct_indices = np.where(correct_mask)[0]
        if len(correct_indices) == 0:
            print(f"No correct samples found for class 1, skipping sample selection for seed {seed}")
            continue
        high_conf_idx = correct_indices[np.argmax(confidence_scores[correct_indices])]
    else:
        high_conf_idx = np.random.choice(high_conf_correct_indices)
    high_conf_score = confidence_scores[high_conf_idx]
    predicted_class = np.argmax(Y_pred_prob[high_conf_idx])
    true_class = np.argmax(Y_test[high_conf_idx])
    print(f"Selected sample {high_conf_idx} with confidence {high_conf_score:.4f} (Predicted: {predicted_class}, True: {true_class})")
    
    # Integrated Gradients voor het geselecteerde sample
    model = load_model(model_file, custom_objects=custom_objects)
    input_data = X_test[high_conf_idx:high_conf_idx+1]
    print(f"input_data.shape: {input_data.shape}, input_data[:5]: {input_data[:5]}")
    baseline = np.mean(X_test, axis=0, keepdims=True)
    ig_attrs = integrated_gradients(model, input_data, baseline=baseline, class_index=1)
    
    # Store data for combined visualization
    seed_data.append((seed, input_data, ig_attrs, high_conf_idx))
    
    # Store results
    results['confidences'].append(Y_pred_prob)
    results['predictions'].append(Y_pred)
    results['true_labels'].append(Y_true)
    results['accuracies'].append(accuracy)
    results['seeds'].append(seed)
    
    # Save predictions
    output_file_dir = os.path.dirname(os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz'))
    os.makedirs(output_file_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz')
    np.savez(
        output_file,
        confidences=Y_pred_prob,
        predictions=Y_pred,
        true_labels=Y_true
    )
    print(f"Saved predictions and confidences to {output_file}")

# Visualize all seeds in one plot
if seed_data:
    visualize_all_attributions_cartopy(
        seed_data, lead, experiment, analysis_output_dir,
        height=24, width=144, channels=2
    )
else:
    print("No valid samples to visualize.")