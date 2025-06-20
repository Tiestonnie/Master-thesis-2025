import numpy as np
import cftime
import xarray as xr
import pandas as pd
import os
from scipy.signal import detrend
import sys

# Paths
input_data_path = "/gpfs/home4/tleneman/Data/Processed_cesm2/"  # Original PSL outputs
output_data_path = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined/"  # New directory for combined outputs
base_path = "/gpfs/home4/tleneman/Data/Processed_cesm2/"
v050_file = os.path.join(base_path, "merged_cesm2_V050.nc")

# Ensure output directory exists
os.makedirs(output_data_path, exist_ok=True)

# Ensemble members
ensemble_indices = list(range(10))  # idx=0 to 9 for 10 ensemble members

# Get array index from SLURM (process one ensemble at a time)
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
ensemble_indices = [idx]

# Lead times
lead_times = [14, 21, 28, 35, 42, 49, 56]

# Detrend function
def detrend_data(data):
    return xr.apply_ufunc(detrend, data, input_core_dims=[['valid_time']], output_core_dims=[['valid_time']])

# Load V050 data (once)
print(f"\nLoading V050 data from {v050_file}")
v050_data = xr.open_dataset(v050_file)
v050_data = v050_data.rename({'V050': 'v050', 'time': 'valid_time'})

# Debug: Check NaNs and coordinates
raw_v050 = v050_data['v050'].values
print("V050 data coordinates:", v050_data.coords)
print("Latitude range:", v050_data['latitude'].values.min(), "to", v050_data['latitude'].values.max())
print("Longitude range:", v050_data['longitude'].values.min(), "to", v050_data['longitude'].values.max())
print(f"NaNs in raw v050 data: {np.isnan(raw_v050).sum()}")

# Convert time to datetime64
time_coord = 'valid_time'
if isinstance(v050_data[time_coord].values[0], (cftime.datetime, cftime.DatetimeNoLeap)):
    dates = pd.to_datetime([d.strftime() for d in v050_data[time_coord].values])
    v050_data[time_coord] = dates
else:
    v050_data[time_coord] = pd.to_datetime(v050_data[time_coord].values, errors='coerce')

if np.all(pd.isna(v050_data[time_coord].values)):
    raise ValueError("All time values are NaT in V050 data.")
print(f"NaNs in V050 time coordinates: {np.sum(pd.isna(v050_data[time_coord].values))}")

# Define 2.5° grid (match PSL)
new_latitude = np.arange(v050_data['latitude'].values.min(), v050_data['latitude'].values.max(), 2.5)
new_longitude = np.arange(v050_data['longitude'].values.min(), v050_data['longitude'].values.max(), 2.5)

# Interpolate V050
v050_data = v050_data.interp(latitude=new_latitude, longitude=new_longitude, method='linear')
nan_count_v050 = np.sum(np.isnan(v050_data['v050'].values))
print(f"NaNs after interpolation (v050): {nan_count_v050}")
if nan_count_v050 > 0:
    print(f"Warning: {nan_count_v050} NaN values in v050 after interpolation. Filling with mean...")
    v050_data['v050'] = v050_data['v050'].fillna(v050_data['v050'].mean(skipna=True))
print("V050 new latitude range:", v050_data['latitude'].values.min(), "to", v050_data['latitude'].values.max())
print("V050 new longitude range:", v050_data['longitude'].values.min(), "to", v050_data['longitude'].values.max())
print("Shape of interpolated v050 data:", v050_data['v050'].shape)

# Compute V050 anomalies
climatology_v050 = v050_data['v050'].mean(dim='valid_time')
X_all_v050 = v050_data['v050'] - climatology_v050
print(f"NaNs after anomaly computation (v050): {np.isnan(X_all_v050.values).sum()}")

# Detrend V050
X_all_v050_detrended = detrend_data(X_all_v050)
print(f"NaNs after detrending (v050): {np.isnan(X_all_v050_detrended.values).sum()}")

# Flatten and standardize V050
X_all_v050_flat = X_all_v050_detrended.stack(z=('latitude', 'longitude')).transpose('valid_time', 'z')
print(f"NaNs after flattening (v050): {np.isnan(X_all_v050_flat.values).sum()}")

X_all_v050_mean = X_all_v050_flat.mean(dim='valid_time')
X_all_v050_std_dev = X_all_v050_flat.std(dim='valid_time')
X_all_v050_std_dev = xr.where(X_all_v050_std_dev == 0, 1, X_all_v050_std_dev)
X_all_v050_normalized = (X_all_v050_flat - X_all_v050_mean) / X_all_v050_std_dev
print(f"NaNs after standardization (v050): {np.isnan(X_all_v050_normalized.values).sum()}")
print(f"X_all_v050_normalized shape: {X_all_v050_normalized.shape}")

# Process each ensemble member
for idx in ensemble_indices:
    print(f"\nProcessing ensemble member {idx}")
    labels_file = os.path.join(input_data_path, f"labels_{idx}.npz")

    # Load labels
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Missing labels file: {labels_file}")
    labels_dict = np.load(labels_file)

    # Process each lead time
    for lead in lead_times:
        # Load preprocessed PSL features
        psl_file = os.path.join(input_data_path, f"member_{idx}_lead_{lead}.npy")
        if not os.path.exists(psl_file):
            raise FileNotFoundError(f"Missing PSL file: {psl_file}")
        X_all_msl_normalized_lead = np.load(psl_file)
        print(f"X_all_msl_normalized_lead shape for lead {lead} days: {X_all_msl_normalized_lead.shape}")

        # Align V050 with valid indices (based on NAO labels)
        Y_all_onehot = labels_dict[f"lead_{lead}"]
        valid_indices = np.arange(Y_all_onehot.shape[0])  # Assume labels dictate valid samples
        try:
            X_all_v050_normalized_lead = X_all_v050_normalized.isel(**{time_coord: valid_indices}).values
        except IndexError:
            raise ValueError(f"Time index mismatch for V050 at ensemble {idx}, lead {lead}. Check time alignment.")
        print(f"X_all_v050_normalized_lead shape for lead {lead} days: {X_all_v050_normalized_lead.shape}")

        # Verify alignment
        if X_all_msl_normalized_lead.shape[0] != X_all_v050_normalized_lead.shape[0]:
            raise ValueError(f"Sample mismatch: PSL has {X_all_msl_normalized_lead.shape[0]}, V050 has {X_all_v050_normalized_lead.shape[0]} samples")

        # Concatenate PSL and V050 features
        X_all_combined_lead = np.concatenate([X_all_msl_normalized_lead, X_all_v050_normalized_lead], axis=1)
        print(f"X_all_combined_lead shape for lead {lead} days: {X_all_combined_lead.shape}")

        # Save combined features to new directory
        output_file = os.path.join(output_data_path, f"member_{idx}_lead_{lead}.npy")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        np.save(output_file, X_all_combined_lead)
        print(f"Saved combined features for lead {lead} to {output_file}")

    print(f"Labels remain unchanged at {labels_file}")