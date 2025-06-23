import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import glob
import pandas as pd
import shap

# Set up logging
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
    nan_percentage = (nan_count / data.size) * 100
    logging.info(f"Loaded {file_path}, shape: {data.shape}, NaNs: {nan_count} ({nan_percentage:.2f}%), Infs: {inf_count}")
    if nan_count > 0 or inf_count > 0:
        logging.warning(f"{nan_count} NaNs and {inf_count} infs in {file_path}")
        if data.ndim == 2:
            column_means = np.nanmean(data, axis=0)
            data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
        else:
            logging.error("NaNs/infs in dataset")
            raise ValueError("NaNs/infs in data. Fix preprocessing.")
    return data

# SHAP function
def compute_shap_values(xgb_model, input_data, feature_names=None, max_samples=1000, seed=None):
    """
    Compute SHAP values for a subset of inputs.
    Args:
        xgb_model: XGBoost model.
        input_data: Input data, shape [n_samples, 6912].
        feature_names: List of feature names for interpretability.
        max_samples: Maximum number of samples to compute SHAP values for.
        seed: Random seed for sampling consistency.
    Returns:
        shap_values: List of [n_samples, 6912] for each class [class_0, class_1].
        input_data_df: DataFrame of input data with feature_names.
    """
    try:
        explainer = shap.TreeExplainer(xgb_model, feature_names=feature_names)
        if input_data.shape[0] > max_samples:
            logging.info(f"Sampling {max_samples} instances from {input_data.shape[0]} for SHAP")
            if seed is not None:
                np.random.seed(seed)
            sample_indices = np.random.choice(input_data.shape[0], max_samples, replace=False)
            input_data_sample = input_data[sample_indices]
        else:
            input_data_sample = input_data
        shap_values = explainer.shap_values(input_data_sample)
        logging.info(f"SHAP values type: {type(shap_values)}, shape: {np.shape(shap_values)}")
        
        # Handle binary classification with single array output
        if isinstance(shap_values, np.ndarray):
            # For binary:logistic, shap_values is for class 1; class 0 is -shap_values
            shap_values = [-shap_values, shap_values]
            logging.info(f"Converted SHAP values to list for binary classification: {[v.shape for v in shap_values]}")
        
        input_data_df = pd.DataFrame(input_data_sample, columns=feature_names)
        return shap_values, input_data_df
    except Exception as e:
        logging.error(f"Error in compute_shap_values: {str(e)}")
        raise

# Generate feature names for 24x144x2 grid
def generate_feature_names(height=24, width=144, channels=['slp', 'v050'], lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0)):
    """Generate feature names for spatial grid."""
    lats = np.linspace(lat_range[0], lat_range[1], height)
    lons = np.linspace(lon_range[0], lon_range[1], width)
    feature_names = []
    for channel in channels:
        for lat in lats:
            for lon in lons:
                feature_names.append(f"{channel}_lat{lat:.1f}_lon{lon:.1f}")
    logging.info(f"Generated {len(feature_names)} feature names")
    return feature_names

# Visualization function
def visualize_attributions_cartopy(attributions, input_data, lead, seed, experiment, sample_idx, output_dir,
                                  height=24, width=144, channels=2, lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0), 
                                  verbose=False, method='SHAP', is_aggregated=False, class_name='positive_NAOI'):
    """
    Visualize SHAP attributions on a Cartopy map, saving as .png for LaTeX.
    Args:
        attributions: SHAP values, shape [6912] for single sample or aggregated.
        input_data: Input data, shape [1, 6912] for single sample, or None for aggregated.
        is_aggregated: If True, visualize mean SHAP values; else, single sample.
        class_name: Class name for title (e.g., 'negative_NAOI').
    """
    try:
        expected_features = height * width * channels
        if attributions.shape[0] != expected_features:
            logging.error(f"Expected attributions shape [{expected_features}], got {attributions.shape}")
            raise ValueError(f"Expected attributions shape [{expected_features}], got {attributions.shape}")
        
        if not is_aggregated:
            if input_data.shape[1] != expected_features:
                logging.error(f"Expected input_data shape [1, {expected_features}], got {input_data.shape}")
                raise ValueError(f"Expected input_data shape [1, {expected_features}], got {input_data.shape}")
            
            if verbose:
                logging.debug(f"Input min: {input_data.min():.2f}, max: {input_data.max():.2f}")
            
            slp_input = input_data[0, :3456].reshape(height, width)
            v050_input = input_data[0, 3456:].reshape(height, width)
        else:
            slp_input = v050_input = None  # No input data for aggregated plot
        
        slp_attrs = attributions[:3456].reshape(height, width)
        v050_attrs = attributions[3456:].reshape(height, width)
        
        # Normalize separately
        max_abs_slp = np.max(np.abs(slp_attrs)) or 1.0
        max_abs_v050 = np.max(np.abs(v050_attrs)) or 1.0
        slp_attrs = slp_attrs / max_abs_slp
        v050_attrs = v050_attrs / max_abs_v050


        shift_indices = 72  # 180° shift
        slp_input = np.roll(slp_input, shift_indices, axis=1) if slp_input is not None else None
        v050_input = np.roll(v050_input, shift_indices, axis=1) if v050_input is not None else None
        slp_attrs = np.roll(slp_attrs, shift_indices, axis=1)
        v050_attrs = np.roll(v050_attrs, shift_indices, axis=1)
        
        lons = np.linspace(lon_range[0], lon_range[1], width)
        lats = np.linspace(lat_range[0], lat_range[1], height)
        
        fig = plt.figure(figsize=(14, 6 if is_aggregated else 8))
        projection = ccrs.PlateCarree()
        
        if not is_aggregated:
            ax1 = fig.add_subplot(221, projection=projection)
            ax1.coastlines()
            ax1.add_feature(cfeature.LAND, facecolor='lightgray')
            ax1.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh1 = ax1.pcolormesh(lons, lats, slp_input, transform=ccrs.PlateCarree(), cmap='coolwarm')
            fig.colorbar(mesh1, ax=ax1, label='SLP (hPa)')
            ax1.set_title(f'SLP Input (Sample {sample_idx})')
            ax1.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
            
            ax2 = fig.add_subplot(222, projection=projection)
            ax2.coastlines()
            ax2.add_feature(cfeature.LAND, facecolor='lightgray')
            ax2.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh2 = ax2.pcolormesh(lons, lats, slp_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            fig.colorbar(mesh2, ax=ax2, label='SLP Attribution')
            ax2.set_title(f'SLP Attributions ({method}, {class_name})')
            ax2.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
            
            ax3 = fig.add_subplot(223, projection=projection)
            ax3.coastlines()
            ax3.add_feature(cfeature.LAND, facecolor='lightgray')
            ax3.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh3 = ax3.pcolormesh(lons, lats, v050_input, transform=ccrs.PlateCarree(), cmap='RdBu_r')
            fig.colorbar(mesh3, ax=ax3, label='v050 (m/s)')
            ax3.set_title(f'v050 Input (Sample {sample_idx})')
            ax3.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
            
            ax4 = fig.add_subplot(224, projection=projection)
            ax4.coastlines()
            ax4.add_feature(cfeature.LAND, facecolor='lightgray')
            ax4.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh4 = ax4.pcolormesh(lons, lats, v050_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            fig.colorbar(mesh4, ax=ax4, label='v050 Attribution')
            ax4.set_title(f'v050 Attributions ({method}, {class_name})')
            ax4.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
        else:
            ax2 = fig.add_subplot(121, projection=projection)
            ax2.coastlines()
            ax2.add_feature(cfeature.LAND, facecolor='lightgray')
            ax2.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh2 = ax2.pcolormesh(lons, lats, slp_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            fig.colorbar(mesh2, ax=ax2, label='SLP SHAP Attribution')
            ax2.set_title(f'SLP Attributions ({method}, {class_name})')
            ax2.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
            
            ax4 = fig.add_subplot(122, projection=projection)
            ax4.coastlines()
            ax4.add_feature(cfeature.LAND, facecolor='lightgray')
            ax4.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
            mesh4 = ax4.pcolormesh(lons, lats, v050_attrs, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
            fig.colorbar(mesh4, ax=ax4, label='v050 SHAP Attribution')
            ax4.set_title(f'v050 Attributions ({method}, {class_name})')
            ax4.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())

        plt.suptitle(f'{method}: Lead {lead}, Seed {seed}, Experiment {experiment}' + 
                     (f', Aggregated {class_name}' if is_aggregated else f', Sample {sample_idx}'))
        plt.tight_layout()
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            if not os.access(output_dir, os.W_OK):
                logging.error(f"No write permission for {output_dir}")
                raise PermissionError(f"No write permission for {output_dir}")
            experiment_safe = experiment.replace('/', '_').replace('.json', '')
            if is_aggregated:
                output_path = os.path.join(output_dir, 
                    f'{method.lower()}_heatmap_cartopy_aggregated_class_0_lead_{lead}_seed_{seed}_{experiment_safe}.png')
            else:
                output_path = os.path.join(output_dir, 
                    f'{method.lower()}_heatmap_cartopy_lead_{lead}_seed_{seed}_sample_{sample_idx}_{experiment_safe}.png')
            plt.savefig(output_path, bbox_inches='tight', format='png', dpi=300)
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
        Y_pred_prob = model.predict(dtest)
        logging.debug(f"Raw Y_pred_prob shape: {Y_pred_prob.shape}")
        if Y_pred_prob.ndim == 1:
            Y_pred_prob = np.stack([1 - Y_pred_prob, Y_pred_prob], axis=1)
        Y_pred = np.argmax(Y_pred_prob, axis=1)
        Y_true = np.argmax(Y_test, axis=1)
        logging.info(f"Y_pred_prob shape: {Y_pred_prob.shape}, sample: {Y_pred_prob[:5]}")
        return Y_pred_prob, Y_pred, Y_true
    except Exception as e:
        logging.error(f"Error in predict_with_model: {str(e)}")
        raise

# Parse arguments
parser = argparse.ArgumentParser(description="Compute SHAP for XGBoost models.")
parser.add_argument('--lead', type=int, required=True, help="Lead time in days")
parser.add_argument('--experiment', type=str, required=True, help="Experiment file path")
args = parser.parse_args()

# Directories
ddir_data = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined"
ddir_out = "/gpfs/home4/tleneman/xgboost_models/"
analysis_output_dir = "/gpfs/home4/tleneman/xgboost_models/results_test_new/analysis/"

# Verify directory
try:
    os.makedirs(analysis_output_dir, exist_ok=True)
    if not os.access(analysis_output_dir, os.W_OK):
        logging.error(f"No write permission for {analysis_output_dir}")
        raise PermissionError(f"No write permission for {analysis_output_dir}")
except Exception as e:
    logging.error(f"Error creating directory {analysis_output_dir}: {str(e)}")
    exit(1)

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']

# Generate feature names
feature_names = generate_feature_names(height=24, width=144, channels=['slp', 'v050'])

# Main loop
lead = args.lead
experiment = args.experiment

# Log available model files
available_models = glob.glob(os.path.join(ddir_out, f"xgb_model_lead_{lead}_seed_*_*"))
logging.info(f"Available model files in {ddir_out}: {available_models}")

# Load test data
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

# Validate input data shape
if X_test.shape[1] != 6912:
    logging.error(f"Unexpected X_test shape: {X_test.shape}")
    raise ValueError(f"Expected X_test shape [n_samples, 6912], got {X_test.shape}")

for seed in seeds:
    experiment_safe = experiment.replace('/', '_').replace('.json', '')
    model_file = os.path.join(ddir_out, f"xgb_model_lead_{lead}_seed_{seed}_{experiment_safe}.json")
    if not os.path.exists(model_file):
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
    
    # Predict for sample selection
    try:
        Y_pred_prob, Y_pred, Y_true = predict_with_model(xgb_model, X_test, Y_test, lead, seed, experiment)
    except Exception as e:
        logging.error(f"Prediction failed for seed {seed}: {str(e)}")
        continue
    
    # Sample selection for high-confidence class 0
    try:
        confidence_scores = Y_pred_prob[:, 1]  # Confidence for class 0
        correct_mask = (Y_pred == Y_true) & (Y_pred == 1) & (Y_pred_prob[:, 1] > 0.772)
        logging.info(f"Correct samples for class 1 with confidence > 0.772: {np.sum(correct_mask)}")
        if np.sum(correct_mask) > 0:
            logging.info(f"Confidence range for class 1: {confidence_scores[correct_mask].min():.4f} to {confidence_scores[correct_mask].max():.4f}")
        high_conf_correct_indices = np.where(correct_mask)[0]
        if len(high_conf_correct_indices) == 0:
            logging.warning(f"No high-confidence samples for class 1 with confidence > 0.622, falling back to correct class 0 samples")
            correct_indices = np.where((Y_pred == Y_true) & (Y_pred == 0))[0]
            if len(correct_indices) == 0:
                logging.warning(f"No correct samples for class 0, selecting random sample")
                high_conf_idx = np.random.choice(np.arange(X_test.shape[0]))
                high_conf_correct_indices = np.array([high_conf_idx])
            else:
                high_conf_idx = correct_indices[np.argmax(confidence_scores[correct_indices])]
                high_conf_correct_indices = np.array([high_conf_idx])
        else:
            high_conf_idx = np.random.choice(high_conf_correct_indices)
        
        logging.info(f"Selected sample {high_conf_idx} with confidence {confidence_scores[high_conf_idx]:.4f}")
        logging.info(f"High-confidence class 0 samples: {len(high_conf_correct_indices)}")
    except Exception as e:
        logging.error(f"Sample selection failed for seed {seed}: {str(e)}")
        continue
    
    input_data = X_test[high_conf_idx:high_conf_idx+1].astype(np.float32)
    
    # SHAP for single sample
    try:
        shap_values_single, _ = compute_shap_values(xgb_model, input_data, feature_names=feature_names, max_samples=1, seed=seed)
        shap_attrs = shap_values_single[0][0]  # Class 0, first sample
        visualize_attributions_cartopy(
            shap_attrs, input_data, lead, seed, experiment, high_conf_idx, analysis_output_dir,
            height=24, width=144, channels=2, verbose=True, method='SHAP', is_aggregated=False, class_name='positive_NAOI'
        )
    except Exception as e:
        logging.error(f"SHAP single-sample computation or visualization failed for seed {seed}: {str(e)}")
    
    # SHAP for high-confidence class 0 samples
    try:
        if len(high_conf_correct_indices) > 0:
            logging.info(f"Computing SHAP values for {len(high_conf_correct_indices)} high-confidence class 0 samples")
            input_data_high_conf = X_test[high_conf_correct_indices].astype(np.float32)
            shap_values, input_data_df = compute_shap_values(
                xgb_model, input_data_high_conf, feature_names=feature_names, max_samples=1000, seed=seed
            )
            
            # Generate aggregated SHAP heatmap for class 0
            try:
                aggregated_shap = shap_values[0].mean(axis=0)  # Mean SHAP for class 0
                logging.info(f"Aggregated SHAP shape: {aggregated_shap.shape}")
                visualize_attributions_cartopy(
                    aggregated_shap, None, lead, seed, experiment, sample_idx='aggregated', 
                    output_dir=analysis_output_dir, height=24, width=144, channels=2, 
                    verbose=False, method='SHAP', is_aggregated=True, class_name='positive_NAOI'
                )
            except Exception as e:
                logging.error(f"Aggregated SHAP visualization failed for seed {seed}: {str(e)}")
            
            # Save SHAP values and generate summary plots for each class
            for class_idx, shap_vals in enumerate(shap_values):  # shap_values = [class_0, class_1]
                class_name = 'negative_NAOI' if class_idx == 0 else 'positive_NAOI'
                logging.info(f"Processing SHAP values for class {class_idx} ({class_name})")
                
                # Save SHAP values as .npy
                shap_output_file = os.path.join(
                    analysis_output_dir, f'shap_values_class_{class_idx}_lead_{lead}_seed_{seed}_{experiment_safe}.npy'
                )
                np.save(shap_output_file, shap_vals)
                logging.info(f"Saved SHAP values for class {class_idx} to {shap_output_file}")
                
                # Save SHAP values as .csv
                shap_df = pd.DataFrame(shap_vals, columns=feature_names)
                shap_csv_file = os.path.join(
                    analysis_output_dir, f'shap_values_class_{class_idx}_lead_{lead}_seed_{seed}_{experiment_safe}.csv'
                )
                shap_df.to_csv(shap_csv_file, index=False)
                logging.info(f"Saved SHAP values for class {class_idx} to {shap_csv_file}")
                
                # SHAP Summary Plot
                plt.figure()
                shap.summary_plot(shap_vals, input_data_df, feature_names=feature_names, show=False, max_display=20)
                summary_plot_file = os.path.join(
                    analysis_output_dir, f'shap_summary_class_{class_idx}_lead_{lead}_seed_{seed}_{experiment_safe}.png'
                )
                plt.savefig(summary_plot_file, bbox_inches='tight')
                plt.close()
                logging.info(f"Saved SHAP summary plot for class {class_idx} to {summary_plot_file}")
        else:
            logging.warning(f"No high-confidence class 0 samples for SHAP analysis, skipping")
    except Exception as e:
        logging.error(f"SHAP subset computation or visualization failed for seed {seed}: {str(e)}")
        continue