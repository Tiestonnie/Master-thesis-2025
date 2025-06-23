import numpy as np
import xgboost as xgb
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import logging
import argparse
import glob
import shap

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='naoi_analysis.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load data
def load_data(file_path):
    """Load and preprocess .npy data, handling NaNs and infinite values."""
    if not os.path.exists(file_path):
        logging.error(f"File {file_path} not found")
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        column_means = np.nanmean(data, axis=0)
        data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
    return data

# SHAP function
def compute_shap_values(xgb_model, input_data, max_samples=1000000, seed=None):
    """Compute SHAP values for a subset of inputs."""
    explainer = shap.TreeExplainer(xgb_model)
    if input_data.shape[0] > max_samples:
        np.random.seed(seed)
        sample_indices = np.random.choice(input_data.shape[0], max_samples, replace=False)
        input_data_sample = input_data[sample_indices]
    else:
        input_data_sample = input_data
    shap_values = explainer.shap_values(input_data_sample)
    if isinstance(shap_values, np.ndarray):
        shap_values = [-shap_values, shap_values]
    return shap_values

# Visualization function
def visualize_attributions_cartopy(attributions_pos, attributions_neg, lead, seed, experiment, output_dir,
                                  height=24, width=144, lat_range=(20.0, 80.0), lon_range=(-180.0, 180.0)):
    """Visualize aggregated SLP attributions for positive and negative NAOI on a single plot with two subplots."""

    # For extracting SLP data
    slp_attrs_pos = attributions_pos[:3456].reshape(height, width)
    slp_attrs_neg = attributions_neg[:3456].reshape(height, width)
    
    # For extracting V050 data
    #slp_attrs_pos = attributions_pos[3456:].reshape(height, width)
    #slp_attrs_neg = attributions_neg[3456:].reshape(height, width)

    # Normalize
    max_abs_slp_pos = np.max(np.abs(slp_attrs_pos)) or 1.0
    max_abs_slp_neg = np.max(np.abs(slp_attrs_neg)) or 1.0
    slp_attrs_pos = slp_attrs_pos / max_abs_slp_pos
    slp_attrs_neg = slp_attrs_neg / max_abs_slp_neg
    
    # Shift longitude
    shift_indices = 72
    slp_attrs_pos = np.roll(slp_attrs_pos, shift_indices, axis=1)
    slp_attrs_neg = np.roll(slp_attrs_neg, shift_indices, axis=1)
    
    lons = np.linspace(lon_range[0], lon_range[1], width)
    lats = np.linspace(lat_range[0], lat_range[1], height)
    
    fig = plt.figure(figsize=(14, 6))
    projection = ccrs.PlateCarree()
    
    # Positive NAOI subplot
    ax1 = fig.add_subplot(121, projection=projection)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
    mesh1 = ax1.pcolormesh(lons, lats, slp_attrs_pos, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    fig.colorbar(mesh1, ax=ax1, label='SLP SHAP Attribution')
    ax1.set_title('Aggregated SLP Attributions (Positive NAOI)')
    ax1.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    # Negative NAOI subplot
    ax2 = fig.add_subplot(122, projection=projection)
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(draw_labels=True, linewidth=0.5, linestyle='--')
    mesh2 = ax2.pcolormesh(lons, lats, slp_attrs_neg, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-0.5, vmax=0.5)
    fig.colorbar(mesh2, ax=ax2, label='SLP SHAP Attribution')
    ax2.set_title('Aggregated SLP Attributions (Negative NAOI)')
    ax2.set_extent([-90, 40, 20, 80], crs=ccrs.PlateCarree())
    
    plt.suptitle(f'SHAP: Lead {lead}, Seed {seed}, Experiment {experiment}')
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    experiment_safe = experiment.replace('/', '_').replace('.json', '')
    output_path = os.path.join(output_dir, f'shap_slp_aggregated_lead_{lead}_seed_{seed}_{experiment_safe}.png')
    plt.savefig(output_path, bbox_inches='tight', format='png', dpi=300)
    plt.close()
    logging.info(f"Saved heatmap to {output_path}")

# Predict with XGBoost
def predict_with_model(model, X_test, Y_test):
    """Predict using XGBoost model."""
    dtest = xgb.DMatrix(X_test)
    Y_pred_prob = model.predict(dtest)
    if Y_pred_prob.ndim == 1:
        Y_pred_prob = np.stack([1 - Y_pred_prob, Y_pred_prob], axis=1)
    Y_pred = np.argmax(Y_pred_prob, axis=1)
    Y_true = np.argmax(Y_test, axis=1)
    return Y_pred_prob, Y_pred, Y_true

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
os.makedirs(analysis_output_dir, exist_ok=True)
if not os.access(analysis_output_dir, os.W_OK):
    logging.error(f"No write permission for {analysis_output_dir}")
    raise PermissionError(f"No write permission for {analysis_output_dir}")

# Parameters
seeds = [410, 133, 33, 210, 47]
test_ens = ['0']

# Main loop
lead = args.lead
experiment = args.experiment

# Load test data
X_test_list = []
Y_test_dict = {}
for ens in test_ens:
    X_file = os.path.join(ddir_data, f'member_{ens}_lead_{lead}.npy')
    if not os.path.exists(X_file):
        logging.error(f"Test file {X_file} not found")
        exit(1)
    X_test_list.append(load_data(X_file))
    
    labels_file = os.path.join(ddir_data, f'labels_{ens}.npz')
    if not os.path.exists(labels_file):
        logging.error(f"Labels file {labels_file} not found")
        exit(1)
    labels_dict = np.load(labels_file)
    lead_key = f'lead_{lead}'
    if lead_key not in labels_dict:
        logging.error(f"Key {lead_key} not found in {labels_file}")
        exit(1)
    Y_test_dict[lead] = labels_dict[lead_key]

X_test = np.concatenate(X_test_list, axis=0)
Y_test = Y_test_dict[lead]
logging.info(f"Test set for lead {lead}: X_test shape {X_test.shape}, Y_test shape {Y_test.shape}")

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
            logging.warning(f"No model file found for seed {seed} at {model_file}")
            continue
    
    logging.info(f"Testing model for lead {lead}, seed {seed}, experiment {experiment}")
    xgb_model = xgb.Booster()
    xgb_model.load_model(model_file)
    
    # Predict
    Y_pred_prob, Y_pred, Y_true = predict_with_model(xgb_model, X_test, Y_test)
    
    # Select high-confidence samples
    confidence_threshold = 0.772
    correct_pos_mask = (Y_pred == Y_true) & (Y_pred == 1) & (Y_pred_prob[:, 1] > confidence_threshold)
    correct_neg_mask = (Y_pred == Y_true) & (Y_pred == 0) & (Y_pred_prob[:, 0] > confidence_threshold)
    pos_indices = np.where(correct_pos_mask)[0]
    neg_indices = np.where(correct_neg_mask)[0]
    
    # Compute SHAP values
    if len(pos_indices) > 0 and len(neg_indices) > 0:
        input_data_pos = X_test[pos_indices].astype(np.float32)
        input_data_neg = X_test[neg_indices].astype(np.float32)
        shap_values_pos = compute_shap_values(xgb_model, input_data_pos, max_samples=1000000, seed=seed)
        shap_values_neg = compute_shap_values(xgb_model, input_data_neg, max_samples=1000000, seed=seed)
        
        # Aggregate SHAP values
        aggregated_shap_pos = shap_values_pos[1].mean(axis=0)  # Positive NAOI (class 1)
        aggregated_shap_neg = shap_values_neg[0].mean(axis=0)  # Negative NAOI (class 0)
        
        # Visualize
        visualize_attributions_cartopy(aggregated_shap_pos, aggregated_shap_neg, lead, seed, experiment, analysis_output_dir)
    else:
        logging.warning(f"No high-confidence samples for one or both classes for seed {seed}")