import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import glob

# Set up logging (DEBUG level for detailed diagnostics)
logging.basicConfig(level=logging.DEBUG, filename='naoi_analysis.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data
def load_data(file_path, subset_size=None):
    """Load and preprocess .npy data, handling NaNs and infinite values."""
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found")
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    if subset_size is not None:
        data = data[:subset_size]
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    inf_count = np.isinf(data).sum()
    logging.info(f"Loaded {file_path}, shape: {data.shape}, NaNs: {nan_count}, Infs: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        logging.warning(f"{nan_count} NaNs and {inf_count} infs in {file_path}")
        if data.ndim == 2:
            column_means = np.nanmean(data, axis=0)
            data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
        else:
            logging.error("NaNs/infs in non-2D data")
            raise ValueError("NaNs/infs in non-2D data. Fix preprocessing.")
    return data

# TensorFlow wrapper for XGBoost to enable gradients
class XGBoostWrapper(tf.keras.Model):
    def __init__(self, xgb_model):
        super(XGBoostWrapper, self).__init__()
        self.xgb_model = xgb_model
    
    def call(self, inputs):
        dmatrix = xgb.DMatrix(inputs.numpy())
        preds = self.xgb_model.predict(dmatrix)
        logging.debug(f"XGBoost predictions shape: {preds.shape}")
        return tf.convert_to_tensor(preds, dtype=tf.float32)

# Integrated Gradients function
def integrated_gradients(model, input_data, baseline=None, num_steps=50, class_index=0):
    """
    Compute Integrated Gradients for a single input.
    Args:
        model: TensorFlow-wrapped XGBoost model.
        input_data: Input data, shape [1, 6912].
        baseline: Reference input (default: mean of dataset).
        num_steps: Number of integration steps.
        class_index: Class index (0=negative NAOI, 1=positive NAOI).
    Returns:
        ig_attrs: Attributions, shape [1, 6912].
    """
    try:
        if baseline is None:
            baseline = np.zeros_like(input_data)
        logging.debug(f"Input data shape: {input_data.shape}, baseline shape: {baseline.shape}")
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
                logging.debug(f"Predictions shape: {predictions.shape}")
                output = predictions[:, class_index]
            grad = tape.gradient(output, inp)
            if grad is None:
                logging.error("Gradient computation failed")
                return np.zeros_like(input_data)
            grads.append(grad)
        avg_grads = tf.reduce_mean(grads, axis=0)
        integrated_grads = (input_data - baseline) * avg_grads
        logging.info(f"IG attrs shape: {integrated_grads.shape}")
        return integrated_grads.numpy()
    except Exception as e:
        logging.error(f"Error in integrated_gradients: {str(e)}")
        raise

# Alternative: SHAP function (recommended for XGBoost)
def compute_shap_values(xgb_model, input_data):
    """
    Compute SHAP values for a single input.
    Args:
        xgb_model: XGBoost model.
        input_data: Input data, shape [1, 6912].
    Returns:
        shap_values: Attributions, shape [1, 6912].
    """
    try:
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(input_data)
        logging.info(f"SHAP values shape: {shap_values.shape}")
        return shap_values  # For binary classification, returns array for positive class
    except Exception as e:
        logging.error(f"Error in compute_shap_values: {str(e)}")
        raise

# Visualization function (optimized)
def visualize_attributions_cartopy(attributions, input_data, lead, seed, experiment, sample_idx, output_dir,
                                  height=24, width=144, channels=2, lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0), verbose=False):
    """
    Visualize attributions on a Cartopy map, saving as .png for LaTeX.
    """
    try:
        expected_features = height * width * channels
        if input_data.shape[1] != expected_features or attributions.shape[1] != expected_features:
            logging.error(f"Expected shape [1, {expected_features}], got {input_data.shape}, {attributions.shape}")
            raise ValueError(f"Expected shape [1, {expected_features}], got {input_data.shape}, {attributions.shape}")
        
        if verbose:
            logging.debug(f"Input min: {input_data.min():.2f}, max: {input_data.max():.2f}")
        
        slp_input = input_data[0, :3456].reshape(height, width)
        v050_input = input_data[0, 3456:].reshape(height, width)
        slp_attrs = attributions[0, :3456].reshape(height, width)
        v050_attrs = attributions[0, 3456:].reshape(height, width)
        
        shift_indices = 72  # 180° shift
        slp_input = np.roll(slp_input, shift_indices, axis=1)
        v050_input = np.roll(v050_input, shift_indices, axis=1)
        slp_attrs = np.roll(slp_attrs, shift_indices, axis=1)
        v050_attrs = np.roll(v050_attrs, shift_indices, axis=1)
        
        logging.info(f"Sample {sample_idx} - SLP min: {slp_input.min():.2f}, max: {slp_input.max():.2f}")
        logging.info(f"Sample {sample_idx} - v050 min: {v050_input.min():.2f}, max: {v050_input.max():.2f}")
        
        # Normaliseer attributies
        slp_attrs = slp_attrs / np.max(np.abs(slp_attrs)) if np.max(np.abs(slp_attrs)) != 0 else slp_attrs
        v050_attrs = v050_attrs / np.max(np.abs(v050_attrs)) if np.max(np.abs(v050_attrs)) != 0 else v050_attrs
        
        lons = np.linspace(lon_range[0], lon_range[1], width)
        #lons = np.roll(lons, shift_indices)  # Align with data shift
        lats = np.linspace(lat_range[0], lat_range[1], height)
        
        fig = plt.figure(figsize=(14, 8))
        projection = ccrs.PlateCarree()
        
        # SLP Input
        ax1 = fig.add_subplot(221, projection=projection)
        ax1.coastlines()
        ax1.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh1 = ax1.pcolormesh(lons, lats, slp_input, transform=ccrs.PlateCarree(), cmap='coolwarm')
        fig.colorbar(mesh1, ax=ax1, label='SLP (hPa)')
        ax1.set_title(f'SLP Input (Sample {sample_idx})')
        ax1.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
        
        # SLP Attributions
        ax2 = fig.add_subplot(222, projection=projection)
        ax2.coastlines()
        ax2.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh2 = ax2.pcolormesh(lons, lats, slp_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(mesh2, ax=ax2, label='Attribution')
        ax2.set_title(f'SLP Attributions (Integrated Gradients)')
        ax2.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
        
        # v050 Input
        ax3 = fig.add_subplot(223, projection=projection)
        ax3.coastlines()
        ax3.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh3 = ax3.pcolormesh(lons, lats, v050_input, transform=ccrs.PlateCarree(), cmap='RdBu_r')
        fig.colorbar(mesh3, ax=ax3, label='v050 (m/s)')
        ax3.set_title(f'v050 Input (Sample {sample_idx})')
        ax3.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
        
        # v050 Attributions
        ax4 = fig.add_subplot(224, projection=projection)
        ax4.coastlines()
        ax4.add_feature(cfeature.LAND, facecolor='lightgray')
        mesh4 = ax4.pcolormesh(lons, lats, v050_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-1, vmax=1)
        fig.colorbar(mesh4, ax=ax4, label='Attribution')
        ax4.set_title(f'v050 Attributions (Integrated Gradients)')
        ax4.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

        plt.suptitle(f'Integrated Gradients: Lead {lead}, Seed {seed}, Experiment {experiment}')
        plt.tight_layout()
        
        # Verify output directory
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                logging.error(f"No write permission for {output_dir}")
                raise PermissionError(f"No write permission for {output_dir}")
            experiment_safe = experiment.replace('/', '_').replace('.json', '')
            output_path = os.path.join(output_dir, f'ig_heatmap_cartopy_lead_{lead}_seed_{seed}_sample_{sample_idx}_{experiment_safe}.png')
            plt.savefig(output_path, bbox_inches='tight', format='png')
            plt.close()
            if os.path.exists(output_path):
                logging.info(f"Saved heatmap to {output_path}")
            else:
                logging.error(f"Failed to save heatmap: {output_path} does not exist")
        except Exception as e:
            logging.error(f"Error saving heatmap to {output_path}: {str(e)}")
            plt.close()
            raise
    except Exception as e:
        logging.error(f"Error in visualize_attributions_cartopy: {str(e)}")
        plt.close()
        raise

# Predict with XGBoost
def predict_with_model(model, X_test, Y_test, lead, seed, experiment):
    """Predict using XGBoost model."""
    try:
        dtest = xgb.DMatrix(X_test)
        Y_pred_prob = model.predict(dtest)  # Shape: [n_samples] or [n_samples, n_classes]
        logging.debug(f"Raw Y_pred_prob shape: {Y_pred_prob.shape}")
        if Y_pred_prob.ndim == 1:  # Binary classification
            Y_pred_prob = np.stack([1 - Y_pred_prob, Y_pred_prob], axis=1)  # Shape: [n_samples, 2]
        Y_pred = np.argmax(Y_pred_prob, axis=1)
        Y_true = np.argmax(Y_test, axis=1)
        accuracy = accuracy_score(Y_true, Y_pred)
        logging.info(f"Lead {lead} days, Seed {seed}, Experiment {experiment} - Test Accuracy: {accuracy:.4f}")
        logging.info(f"Y_pred_prob shape: {Y_pred_prob.shape}, sample: {Y_pred_prob[:5]}")
        return Y_pred_prob, Y_pred, Y_true, accuracy
    except Exception as e:
        logging.error(f"Error in predict_with_model: {str(e)}")
        raise

# Parse arguments
parser = argparse.ArgumentParser(description="Run XGBoost model testing.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined"
ddir_out = "/gpfs/home4/tleneman/xgboost_models/"
output_dir = "/gpfs/home4/tleneman/xgboost_models/results_test_set/"
analysis_output_dir = "/gpfs/home4/tleneman/xgboost_models/results_test_new/analysis/"

# Verify directories
for dir_path in [output_dir, analysis_output_dir]:
    try:
        os.makedirs(dir_path, exist_ok=True)
        if not os.access(dir_path, os.W_OK):
            logging.error(f"No write permission for {dir_path}")
            raise PermissionError(f"No write permission for {dir_path}")
    except Exception as e:
        logging.error(f"Error creating directory {dir_path}: {str(e)}")
        exit(1)

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']

# Main loop
lead = args.lead
experiment = args.experiment
results = {'confidences': [], 'predictions': [], 'true_labels': [], 'accuracies': [], 'seeds': []}

# Log available model files
available_models = glob.glob(os.path.join(ddir_out, f"xgb_model_lead_{lead}_seed_*_*"))
logging.info(f"Available model files in {ddir_out}: {available_models}")

X_test_list = []
Y_test_dict = {}
for ens in test_ens:
    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    if not os.path.exists(X_file):
        logging.error(f"Test file {X_file} not found")
        exit(1)
    X = load_data(X_file)
    X_test_list.append(X)
    
    labels_file = os.path.join(ddir_data, f'labels_{ens}.npz')
    if not os.path.exists(labels_file):
        logging.error(f"Labels file {labels_file} not found")
        exit(1)
    labels_dict = np.load(labels_file)
    lead_key = f'lead_{lead}'
    if lead_key not in labels_dict:
        logging.error(f"Key {lead_key} not found in {labels_file}")
        exit(1)
    Y = labels_dict[lead_key]
    Y_test_dict[lead] = Y

if not X_test_list:
    logging.error(f"No test data loaded for lead {lead}")
    exit(1)

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]
logging.info(f"Test set for lead {lead}: X_test shape {X_test.shape}, Y_test shape {Y_test.shape}")

for seed in seeds:
    # Try correct extension (.json) and handle experiment name
    experiment_safe = experiment.replace('/', '_').replace('.json', '')  # Convert exp_1/exp_100.json to exp_1_exp_100
    model_file = os.path.join(ddir_out, f"xgb_model_lead_{lead}_seed_{seed}_{experiment_safe}.json")
    if not os.path.exists(model_file):
        # Try alternative extensions
        for ext in ['.model', '.pkl']:
            alt_model_file = os.path.join(ddir_out, f"xgb_model_lead_{lead}_seed_{seed}_{experiment_safe}{ext}")
            if os.path.exists(alt_model_file):
                model_file = alt_model_file
                break
        else:
            logging.warning(f"No model file found for seed {seed} at {model_file} or with extensions .model, .pkl")
            continue
    
    logging.info(f"Testing model for lead {lead}, seed {seed}, experiment {experiment}, file {model_file}")
    xgb_model = xgb.Booster()
    try:
        xgb_model.load_model(model_file)
    except Exception as e:
        logging.error(f"Failed to load model {model_file}: {str(e)}")
        continue
    
    # Predict
    try:
        Y_pred_prob, Y_pred, Y_true, accuracy = predict_with_model(xgb_model, X_test, Y_test, lead, seed, experiment)
    except Exception as e:
        logging.error(f"Prediction failed for seed {seed}: {str(e)}")
        continue
    
    # Sample selection
    try:
        confidence_scores = Y_pred_prob[:, 0] if Y_pred_prob.ndim == 2 else Y_pred_prob
        correct_mask = (Y_pred == Y_true) & (Y_pred == 0)
        logging.info(f"Correct samples for class 0: {np.sum(correct_mask)}")
        if np.sum(correct_mask) > 0:
            logging.info(f"Confidence range: {confidence_scores[correct_mask].min():.4f} to {confidence_scores[correct_mask].max():.4f}")
        threshold = np.percentile(confidence_scores[correct_mask], 90) if np.any(correct_mask) else 0.5
        high_conf_correct_indices = np.where(correct_mask & (confidence_scores > threshold))[0]
        if len(high_conf_correct_indices) == 0:
            logging.warning(f"No high-confidence samples for class 0, falling back")
            correct_indices = np.where(correct_mask)[0]
            if len(correct_indices) == 0:
                logging.warning(f"No correct samples for class 0, selecting random sample")
                high_conf_idx = np.random.choice(np.arange(X_test.shape[0]))
            else:
                high_conf_idx = correct_indices[np.argmax(confidence_scores[correct_indices])]
        else:
            high_conf_idx = np.random.choice(high_conf_correct_indices)
        
        if high_conf_idx >= X_test.shape[0]:
            logging.error(f"Sample index {high_conf_idx} exceeds X_test size {X_test.shape[0]}")
            continue
        
        logging.info(f"Selected sample {high_conf_idx} with confidence {confidence_scores[high_conf_idx]:.4f}")
    except Exception as e:
        logging.error(f"Sample selection failed for seed {seed}: {str(e)}")
        continue
    
    # Integrated Gradients
    input_data = X_test[high_conf_idx:high_conf_idx+1].astype(np.float32)
    try:
        wrapped_model = XGBoostWrapper(xgb_model)
        baseline = np.mean(X_test, axis=0, keepdims=True).astype(np.float32)
        class_index = 0
        ig_attrs = integrated_gradients(wrapped_model, input_data, baseline=baseline, class_index=class_index)
    except Exception as e:
        logging.error(f"IG computation failed for seed {seed}: {str(e)}")
        # Fallback to SHAP
        try:
            ig_attrs = compute_shap_values(xgb_model, input_data)
        except Exception as e:
            logging.error(f"SHAP computation failed for seed {seed}: {str(e)}")
            continue
    
    # Visualize
    try:
        visualize_attributions_cartopy(
            ig_attrs, input_data, lead, seed, experiment, high_conf_idx, analysis_output_dir,
            height=24, width=144, channels=2, verbose=True
        )
    except Exception as e:
        logging.error(f"Visualization failed for seed {seed}: {str(e)}")
        continue
    
    # Save results
    results['confidences'].append(Y_pred_prob)
    results['predictions'].append(Y_pred)
    results['true_labels'].append(Y_true)
    results['accuracies'].append(accuracy)
    results['seeds'].append(seed)
    
    output_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment_safe}_new.npz')
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.savez(output_file, confidences=Y_pred_prob, predictions=Y_pred, true_labels=Y_true)
        logging.info(f"Saved predictions to {output_file}")
    except Exception as e:
        logging.error(f"Error saving predictions to {output_file}: {str(e)}")

# Analysis phase
accuracy_df = pd.DataFrame(columns=['lead', 'experiment', 'seed', 'bin_1_accuracy', 'bin_2_accuracy', 'bin_3_accuracy',
                                    'bin_4_accuracy', 'bin_5_accuracy', 'bin_6_accuracy', 'bin_7_accuracy',
                                    'bin_8_accuracy', 'bin_9_accuracy', 'bin_10_accuracy', 'overall_accuracy'])

for seed in seeds:
    experiment_safe = experiment.replace('/', '_').replace('.json', '')
    npz_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment_safe}_new.npz')
    if not os.path.exists(npz_file):
        logging.warning(f"Results file {npz_file} not found, skipping seed {seed}")
        print(f"Results file {npz_file} not found, skipping...")
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
        csv_file = os.path.join(analysis_output_dir, f'sorted_predictions_lead_{lead}_seed_{seed}_{experiment_safe}_new.csv')
        df_sorted.to_csv(csv_file, index=False)
        logging.info(f"Saved sorted predictions to {csv_file}")
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
        logging.info(f"Added row to accuracy_df: {row_data}")
        print(f"Added row to accuracy_df: {row_data}")
        
        plt.figure(figsize=(8, 6))
        plt.hist(confidence_scores, bins=20, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution: Lead {lead}, Seed {seed}, Experiment {experiment}')
        hist_path = os.path.join(analysis_output_dir, f'confidence_histogram_lead_{lead}_seed_{seed}_{experiment_safe}_new.png')
        plt.savefig(hist_path)
        plt.close()
        logging.info(f"Saved histogram to {hist_path}")
    except Exception as e:
        logging.error(f"Error in analysis for seed {seed}: {str(e)}")
        continue

# Append to CSV
try:
    accuracy_csv = os.path.join(analysis_output_dir, 'accuracy_by_confidence_bins.csv')
    if not os.path.exists(accuracy_csv):
        accuracy_df.to_csv(accuracy_csv, index=False)
    else:
        accuracy_df.to_csv(accuracy_csv, mode='a', header=False, index=False)
    logging.info(f"Saved accuracy by confidence bins to {accuracy_csv} with {len(accuracy_df)} rows")
    print(f"Saved accuracy by confidence bins to {accuracy_csv} with {len(accuracy_df)} rows")
except Exception as e:
    logging.error(f"Error saving accuracy CSV: {str(e)}")

# Plot accuracy by confidence bins
try:
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
        plot_path = os.path.join(analysis_output_dir, 'accuracy_by_confidence_bins.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved accuracy plot to {plot_path}")
    else:
        logging.warning("accuracy_df is empty, skipping final plot")
        print("Warning: accuracy_df is empty, skipping final plot.")
except Exception as e:
    logging.error(f"Error in plotting accuracy by confidence bins: {str(e)}")