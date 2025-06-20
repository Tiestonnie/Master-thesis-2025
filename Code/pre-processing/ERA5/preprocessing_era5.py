import numpy as np
import cftime
import xarray as xr
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from scipy.signal import detrend
import sys

# Path for processed data
processed_data_path = "/gpfs/home4/tleneman/Data/Processed_era5/"
base_path = "/gpfs/home4/tleneman/Data/Processed_era5/"

# Ensure output directory exists
os.makedirs(processed_data_path, exist_ok=True)

# Dataset files
dataset_files = ["merged_nao_data_1940-2025.nc"]

# Get array index from SLURM
idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
dataset_files = [dataset_files[idx]]

# Lead times
lead_times = [14, 21, 28, 35, 42, 49, 56]

# Detrend function
def detrend_data(data):
    return xr.apply_ufunc(detrend, data, input_core_dims=[['valid_time']], output_core_dims=[['valid_time']])

# Process one dataset
for file in dataset_files:
    member_file = os.path.join(processed_data_path, f"member_{idx}.npy")
    labels_file = os.path.join(processed_data_path, f"labels_{idx}.npz")

    print(f"\nProcessing {file}")
    nao_data = xr.open_dataset(os.path.join(base_path, file))
#    nao_data = nao_data.rename({'PSL': 'msl', 'time': 'valid_time'})
    
    # Debug: Check NaNs in raw data
    raw_msl = nao_data['msl'].values
    print("nao_data coordinates:", nao_data.coords)
    print("Latitude range:", nao_data['latitude'].values.min(), "to", nao_data['latitude'].values.max())
    print("Longitude range:", nao_data['longitude'].values.min(), "to", nao_data['longitude'].values.max())
    print(f"NaNs in raw msl data: {np.isnan(raw_msl).sum()}")

    # Convert time to datetime64 format
    time_coord = 'valid_time'
    if isinstance(nao_data[time_coord].values[0], (cftime.datetime, cftime.DatetimeNoLeap)):
        dates = pd.to_datetime([d.strftime() for d in nao_data[time_coord].values])
        nao_data[time_coord] = dates
    else:
        nao_data[time_coord] = pd.to_datetime(nao_data[time_coord].values, errors='coerce')
    
    # Check for NaT values
    if np.all(pd.isna(nao_data[time_coord].values)):
        raise ValueError("All time values are NaT. Check time decoding in earlier steps or dataset structure.")
    print(f"NaNs in time coordinates: {np.sum(pd.isna(nao_data[time_coord].values))}")

    # Define new 2.5° grid within original bounds
    new_latitude = np.arange(nao_data['latitude'].values.min(), nao_data['latitude'].values.max(), 2.5)
    new_longitude = np.arange(nao_data['longitude'].values.min(), nao_data['longitude'].values.max(), 2.5)
    
    # Interpolate the data
    nao_data = nao_data.interp(latitude=new_latitude, longitude=new_longitude, method='linear')
    nan_count_interp = np.sum(np.isnan(nao_data['msl'].values))
    print(f"NaNs after interpolation: {nan_count_interp}")
    if nan_count_interp > 0:
        print(f"Warning: {nan_count_interp} nan values after interpolation. Filling with mean...")
        nao_data['msl'] = nao_data['msl'].fillna(nao_data['msl'].mean(skipna=True))
    print("Nieuw latitude bereik:", nao_data['latitude'].values.min(), "to", nao_data['latitude'].values.max())
    print("Nieuw longitude bereik:", nao_data['longitude'].values.min(), "to", nao_data['longitude'].values.max())
    print("Vorm van de geinterpoleerde data:", nao_data['msl'].shape)

    # Compute anomalies
    climatology = nao_data['msl'].mean(dim='valid_time')
    X_all = nao_data['msl'] - climatology
    print(f"NaNs after anomaly computation: {np.isnan(X_all.values).sum()}")

    # Detrend
    X_all_detrended = detrend_data(X_all)
    print(f"NaNs after detrending: {np.isnan(X_all_detrended.values).sum()}")

    # Flatten and standardize
    X_all_flat = X_all_detrended.stack(z=('latitude', 'longitude')).transpose('valid_time', 'z')
    print(f"NaNs after flattening: {np.isnan(X_all_flat.values).sum()}")
    X_all_mean = X_all_flat.mean(dim='valid_time')
    X_all_std_dev = X_all_flat.std(dim='valid_time')
    X_all_std_dev = xr.where(X_all_std_dev == 0, 1, X_all_std_dev)
    X_all_normalized = (X_all_flat - X_all_mean) / X_all_std_dev
    print(f"NaNs after standardization: {np.isnan(X_all_normalized.values).sum()}")
    print(f"X_all_normalized shape voor {file}: {X_all_normalized.shape}")

    # Define NAO regions
    iceland = nao_data.sel(latitude=slice(65, 70), longitude=slice(340, 350))
    azores = nao_data.sel(latitude=slice(35, 40), longitude=slice(325, 335))
    
    # Compute means over regions
    iceland = iceland.mean(dim=['latitude', 'longitude'])
    azores = azores.mean(dim=['latitude', 'longitude'])
    print(f"NaNs in iceland region: {np.isnan(iceland['msl'].values).sum()}")
    print(f"NaNs in azores region: {np.isnan(azores['msl'].values).sum()}")

    # Compute NAO index
    nao_index = azores['msl'] - iceland['msl']
    nao_index = nao_index.rename('nao_index')
    print("NAO Index shape:", nao_index.shape)
    print("NAO Index values:", nao_index.values[:3], "...", nao_index.values[-3:])
    print(f"NaNs in NAO Index: {np.isnan(nao_index.values).sum()}")

    days_average = 3

    # Nieuwe code: Bereken gemeenschappelijke valid_indices en verwerk X en Y
    all_valid_indices = []
    for lead in lead_times:
        shift_steps = lead
        nao_index_shifted = nao_index.roll(valid_time=-shift_steps, roll_coords=True)
        nao_index_shifted[-shift_steps:] = np.nan
        Y_all = nao_index_shifted.rolling(**{time_coord: days_average}, center=True, min_periods=1).mean()
        valid_indices = np.where(~np.isnan(Y_all))[0]
        all_valid_indices.append(valid_indices)

    common_valid_indices = all_valid_indices[0]
    for indices in all_valid_indices[1:]:
        common_valid_indices = np.intersect1d(common_valid_indices, indices)
    print(f"Common valid indices across all lead times: {len(common_valid_indices)}")

    Y_members = {}
    X_members = {}
    for lead in lead_times:
        shift_steps = lead
        nao_index_shifted = nao_index.roll(valid_time=-shift_steps, roll_coords=True)
        nao_index_shifted[-shift_steps:] = np.nan
        Y_all = nao_index_shifted.rolling(**{time_coord: days_average}, center=True, min_periods=1).mean()
        
        Y_all = Y_all.isel(**{time_coord: common_valid_indices})
        X_all_normalized_lead = X_all_normalized.isel(**{time_coord: common_valid_indices})
        
        print(f"Y_all shape na NaN-verwijdering voor lead {lead} dagen: {Y_all.shape}")
        print(f"X_all_normalized_lead shape na NaN-verwijdering voor lead {lead} dagen: {X_all_normalized_lead.shape}")
        
        Ytrain_med = np.median(Y_all.values)
        Y_all_binary = xr.where(Y_all >= Ytrain_med, 1, 0)
        enc = OneHotEncoder(sparse_output=False)
        Y_all_onehot = enc.fit_transform(Y_all_binary.values.reshape(-1, 1))
        print(f"Y_all_onehot shape voor lead {lead} dagen: {Y_all_onehot.shape}")
        
        Y_members[f"lead_{lead}"] = Y_all_onehot
        X_members[f"lead_{lead}"] = X_all_normalized_lead.values

    # Controleer vormen en NaN's
    for lead in lead_times:
        print(f"Final X_members[lead_{lead}] shape: {X_members[f'lead_{lead}'].shape}")
        print(f"Final Y_members[lead_{lead}] shape: {Y_members[f'lead_{lead}'].shape}")
        print(f"NaNs in final X_members[lead_{lead}]: {np.sum(np.isnan(X_members[f'lead_{lead}']))}")

# Sla gegevens op
for lead in lead_times:
    member_file = os.path.join(processed_data_path, f"3day_member_{idx}_lead_{lead}.npy")
    np.save(member_file, X_members[f"lead_{lead}"])
np.savez(labels_file, **Y_members)
print(f"Verwerkte data en labels voor {file} opgeslagen in {processed_data_path}")

# Save X_all_normalized for common valid indices as member_{idx}.npy
member_file = os.path.join(processed_data_path, f"member_{idx}.npy")
X_member = X_all_normalized.isel(valid_time=common_valid_indices).values
np.save(member_file, X_member)
print(f"Input data saved as {member_file} with shape {X_member.shape}")