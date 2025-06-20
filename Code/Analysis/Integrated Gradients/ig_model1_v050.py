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
    
    # Interpoleer tussen baseline en input
    interpolated_inputs = [
        baseline + (float(i) / num_steps) * (input_data - baseline)
        for i in range(num_steps + 1)
    ]
    
    # Converteer naar tensor
    interpolated_inputs = tf.convert_to_tensor(interpolated_inputs, dtype=tf.float32)
    
    # Bereken gradiënten
    grads = []
    for inp in interpolated_inputs:
        with tf.GradientTape() as tape:
            tape.watch(inp)
            predictions = model(inp)
            output = predictions[:, class_index]  # Focus op de gewenste klasse
        grad = tape.gradient(output, inp)
        grads.append(grad)
    
    # Gemiddelde gradiënten
    avg_grads = tf.reduce_mean(grads, axis=0)
    
    # Bereken Integrated Gradients: (input - baseline) * gemiddelde gradiënten
    integrated_grads = (input_data - baseline) * avg_grads
    return integrated_grads.numpy()

def visualize_attributions_cartopy(attributions, input_data, lead, seed, experiment, sample_idx, output_dir, height=24, width=144, channels=2, lat_range=(20.0, 80.0), lon_range=(-90.0, 40.0)):
    """
    Visualiseer Integrated Gradients attributies op een kaart.
    Args:
        attributions: Attributies met shape [1, 6912].
        input_data: Originele input met shape [1, 6912].
        lead: Lead time in dagen.
        seed: Model seed.
        experiment: Experiment naam.
        sample_idx: Index van het testvoorbeeld.
        output_dir: Directory om heatmaps op te slaan.
        height: Hoogte van de grid (24 breedtegraden).
        width: Breedte van de grid (144 lengtegraden).
        channels: Aantal kanalen (2 voor SLP en v050).
        lat_range: Tuple van (min_lat, max_lat).
        lon_range: Tuple van (min_lon, max_lon).
    """
    # Reshape input en attributies
    input_reshaped = input_data.reshape(1, height, width, channels)
    attrs_reshaped = attributions.reshape(1, height, width, channels)
    
    # Extraheer SLP en v050
    slp_input = input_reshaped[0, :, :, 0]  # Shape: [24, 144]
    v050_input = input_reshaped[0, :, :, 1]  # Shape: [24, 144]
    slp_attrs = attrs_reshaped[0, :, :, 0]  # Shape: [24, 144]
    v050_attrs = attrs_reshaped[0, :, :, 1]  # Shape: [24, 144]
    
    # Controleer kanaalvolgorde door bereik te printen
    print(f"Sample {sample_idx} - SLP kanaal (kanaal 0) min: {slp_input.min():.2f}, max: {slp_input.max():.2f}")
    print(f"Sample {sample_idx} - v050 kanaal (kanaal 1) min: {v050_input.min():.2f}, max: {v050_input.max():.2f}")
    
    # Normaliseer attributies
    slp_attrs = slp_attrs / np.max(np.abs(slp_attrs)) if np.max(np.abs(slp_attrs)) != 0 else slp_attrs
    v050_attrs = v050_attrs / np.max(np.abs(v050_attrs)) if np.max(np.abs(v050_attrs)) != 0 else v050_attrs
    
    # Definieer coördinaten
   
    lons = np.linspace(lon_range[0], lon_range[1], width)
    lats = np.linspace(lat_range[1], lat_range[0], height)
    
    # Maak een figuur met 2x2 subplots
    fig = plt.figure(figsize=(14, 8))
    projection = ccrs.PlateCarree()
    
    # Plot SLP input
    ax1 = fig.add_subplot(221, projection=projection)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh1 = ax1.pcolormesh(lons, lats, slp_input, transform=ccrs.PlateCarree(), cmap='coolwarm')
    fig.colorbar(mesh1, ax=ax1, label='SLP (hPa)')
    ax1.set_title(f'SLP Input (Sample {sample_idx})')
    ax1.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    # Plot SLP attributies
    ax2 = fig.add_subplot(222, projection=projection)
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh2 = ax2.pcolormesh(lons, lats, np.abs(slp_attrs), transform=ccrs.PlateCarree(), cmap='Reds')
    fig.colorbar(mesh2, ax=ax2, label='Attribution')
    ax2.set_title(f'SLP Attributions (Integrated Gradients)')
    ax2.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    # Plot v050 input
    ax3 = fig.add_subplot(223, projection=projection)
    ax3.coastlines()
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh3 = ax3.pcolormesh(lons, lats, v050_input, transform=ccrs.PlateCarree(), cmap='coolwarm')
    fig.colorbar(mesh3, ax=ax3, label='v050 (m/s)')
    ax3.set_title(f'v050 Input (Sample {sample_idx})')
    ax3.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    # Plot v050 attributies
    ax4 = fig.add_subplot(224, projection=projection)
    ax4.coastlines()
    ax4.add_feature(cfeature.LAND, facecolor='lightgray')
    mesh4 = ax4.pcolormesh(lons, lats, np.abs(v050_attrs), transform=ccrs.PlateCarree(), cmap='Reds')
    fig.colorbar(mesh4, ax=ax4, label='Attribution')
    ax4.set_title(f'v050 Attributions (Integrated Gradients)')
    ax4.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    plt.suptitle(f'Integrated Gradients: Lead {lead}, Seed {seed}, Experiment {experiment}')
    plt.tight_layout()
    
    # Sla de figuur op
    os.makedirs(output_dir, exist_ok=True)
    experiment = experiment.replace('/', '_')  # Vervang '/' door '_' in experiment naam
    output_path = os.path.join(output_dir, f'ig_heatmap_cartopy_lead_{lead}_seed_{seed}_sample_{sample_idx}_{experiment.replace(".h5", "")}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Cartopy Integrated Gradients heatmap to {output_path}")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Run model testing for a specific lead time and experiment.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2/test_set_v050/"
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

# Main testing loop for the specific lead time and experiment
lead = args.lead
experiment = args.experiment
results = {'confidences': [], 'predictions': [], 'true_labels': [], 'accuracies': [], 'seeds': []}

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
    
    # Selecteer een willekeurige hoge-confidence sample voor klasse 1 (positief NAO)
    confidence_scores = Y_pred_prob[:, 1]  # Confidence voor klasse 1
    high_conf_indices = np.where(confidence_scores > 0.7)[0]  # Samples met confidence > 0.7
    if len(high_conf_indices) == 0:
        high_conf_idx = np.argmax(confidence_scores)  # Fallback: hoogste confidence
    else:
        high_conf_idx = np.random.choice(high_conf_indices)  # Willekeurige keuze uit hoge-confidence samples
    high_conf_score = confidence_scores[high_conf_idx]
    predicted_class = np.argmax(Y_pred_prob[high_conf_idx])
    true_class = np.argmax(Y_test[high_conf_idx])
    print(f"Selected sample {high_conf_idx} with confidence {high_conf_score:.4f} (Predicted: {predicted_class}, True: {true_class})")
    
    # Integrated Gradients voor het geselecteerde sample
    model = load_model(model_file, custom_objects=custom_objects)
    input_data = X_test[high_conf_idx:high_conf_idx+1]  # Shape: [1, 6912]
    baseline = np.mean(X_test, axis=0, keepdims=True)  # Shape: [1, 6912]
    ig_attrs = integrated_gradients(model, input_data, baseline=baseline, class_index=1)
    visualize_attributions_cartopy(
        ig_attrs, input_data, lead, seed, experiment, high_conf_idx, analysis_output_dir,
        height=24, width=144, channels=2, lat_range=(20.0, 80.0), lon_range=(-90.0, 40.0)
    )
    
    # Rest van je bestaande code
    results['confidences'].append(Y_pred_prob)
    results['predictions'].append(Y_pred)
    results['true_labels'].append(Y_true)
    results['accuracies'].append(accuracy)
    results['seeds'].append(seed)
    
    # Create the full directory path for the output file
    output_file_dir = os.path.dirname(os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz'))
    os.makedirs(output_file_dir, exist_ok=True)
    
    # Save the predictions
    output_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz')
    np.savez(
        output_file,
        confidences=Y_pred_prob,
        predictions=Y_pred,
        true_labels=Y_true
    )
    print(f"Saved predictions and confidences to {output_file}")

# --- Analysis and Visualization Phase ---
# (Jouw originele analysecode blijft ongewijzigd)
accuracy_df = pd.DataFrame(columns=['lead', 'experiment', 'seed', 'bin_1_accuracy', 'bin_2_accuracy', 'bin_3_accuracy', 
                                    'bin_4_accuracy', 'bin_5_accuracy', 'bin_6_accuracy', 'bin_7_accuracy', 
                                    'bin_8_accuracy', 'bin_9_accuracy', 'bin_10_accuracy', 'overall_accuracy'])

for seed in seeds:
    npz_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.npz')
    if not os.path.exists(npz_file):
        print(f"Results file {npz_file} not found, skipping...")
        continue
    
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
    csv_file = os.path.join(analysis_output_dir, f'sorted_predictions_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.csv')
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
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Bin {i+1} (Confidence {bin_df['confidence_score'].min() if not bin_df.empty else 0:.4f} to {bin_df['confidence_score'].max() if not bin_df.empty else 0:.4f}): Accuracy = {bin_accuracy:.4f}")
    
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
        'overall_accuracy': overall_accuracy
    }
    row_df = pd.DataFrame([row_data])
    accuracy_df = pd.concat([accuracy_df, row_df], ignore_index=True)
    print(f"Added row to accuracy_df: {row_data}")
    
    plt.figure(figsize=(8, 6))
    plt.hist(confidence_scores, bins=20, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title(f'Confidence Distribution: Lead {lead}, Seed {seed}, Experiment {experiment}')
    plt.savefig(os.path.join(analysis_output_dir, f'confidence_histogram_lead_{lead}_seed_{seed}_{experiment.replace(".h5", "")}_new.png'))
    plt.close()

# Append to CSV instead of overwriting
accuracy_csv = os.path.join(analysis_output_dir, 'accuracy_by_confidence_bins.csv')
if not os.path.exists(accuracy_csv):
    accuracy_df.to_csv(accuracy_csv, index=False)
else:
    accuracy_df.to_csv(accuracy_csv, mode='a', header=False, index=False)
print(f"Saved accuracy by confidence bins to {accuracy_csv} with {len(accuracy_df)} rows")

plt.figure(figsize=(12, 6))
if not accuracy_df.empty:
    for index, row in accuracy_df.iterrows():
        accuracies = [row['bin_1_accuracy'], row['bin_2_accuracy'], row['bin_3_accuracy'], row['bin_4_accuracy'],
                      row['bin_5_accuracy'], row['bin_6_accuracy'], row['bin_7_accuracy'], row['bin_8_accuracy'],
                      row['bin_9_accuracy'], row['bin_10_accuracy']]
        plt.plot(range(1, 11), accuracies, marker='o', label=f'Lead {row["lead"]} (Exp {row["experiment"]}, Seed {row["seed"]})')
    plt.xlabel('Confidence Bin (1=Least Confident, 10=Most Confident)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Confidence Bin Across Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_output_dir, 'accuracy_by_confidence_bins.png'))
    plt.close()
else:
    print("Warning: accuracy_df is empty, skipping final plot.")