import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error

# --- Configuration ---

# Directories CESM2
#output_dir = "/gpfs/home4/tleneman/model1/results_test_set/"  # Where .npz files are stored
#analysis_output_dir = "/gpfs/home4/tleneman/model1/results_test_new/analysis/"  # Where results will be saved
#os.makedirs(analysis_output_dir, exist_ok=True)
#labels_file = "/gpfs/home4/tleneman/Data/Processed_cesm2/test_set/labels_0.npz"

# Directories CESM2
output_dir = "/gpfs/home4/tleneman/model1/results_test_set_era5/"  # Where .npz files are stored
analysis_output_dir = "/gpfs/home4/tleneman/model1/results_test_era5/analysis/"  # Where results will be saved
os.makedirs(analysis_output_dir, exist_ok=True)
labels_file = "/gpfs/home4/tleneman/Data/Processed_era5/14 day/labels_0.npz"


labels_dict = np.load(labels_file)
print("Labels file keys:", labels_dict.files)  # Debug: List all keys
print("Sample lead_14:", labels_dict['lead_14'][:20])  # Debug: Check more values

# Parameters
experiments = [
    'exp_1/exp_100',  # lead=14
    'exp_4/exp_400',  # lead=21
    'exp_5/exp_500',  # lead=28
    'exp_6/exp_600',  # lead=35
    'exp_7/exp_700',  # lead=42
    'exp_8/exp_800',  # lead=49
    'exp_9/exp_900',  # lead=56
]
lead_times = [14, 21, 28, 35, 42, 49, 56]
seeds = [99, 133, 210, 33, 410, 47, 64, 692, 910, 92]

# --- Analysis and Visualization Phase ---

# Initialize DataFrame for accuracy, RMSE, and correlations
accuracy_df = pd.DataFrame(columns=['lead', 'experiment', 'seed', 'bin_1_accuracy', 'bin_2_accuracy', 'bin_3_accuracy', 
                                    'bin_4_accuracy', 'bin_5_accuracy', 'bin_6_accuracy', 'bin_7_accuracy', 
                                    'bin_8_accuracy', 'bin_9_accuracy', 'bin_10_accuracy', 'overall_accuracy',
                                    'rmse', 'correlation_pos', 'correlation_neg', 'final_correlation', 'overall_correlation'])

# Process each model's results
for lead_idx, (experiment, lead) in enumerate(zip(experiments, lead_times)):
    for seed in seeds:
        npz_file = os.path.join(output_dir, f'predictions_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.npz')
        if not os.path.exists(npz_file):
            print(f"Results file {npz_file} not found, skipping...")
            continue
        
        # Load results
        data = np.load(npz_file)
        confidences = data['confidences']
        predictions = data['predictions']
        true_labels_npz = data['true_labels']  # Original true labels from .npz
        print(f"Lead {lead}, Seed {seed}, True Labels (NPZ) shape: {true_labels_npz.shape}, Sample: {true_labels_npz[:20]}, Unique values: {np.unique(true_labels_npz, return_counts=True)}")  # Debug
        print(f"Lead {lead}, Seed {seed}, Confidences sample: {confidences[:5]}, Unique max confidences: {np.unique(np.max(confidences, axis=1), return_counts=True)}")  # Debug

        # Use labels_0.npz as true labels
        if f'lead_{lead}' in labels_dict:
            true_naoi = labels_dict[f'lead_{lead}']
            print(f"Lead {lead}, Seed {seed}, True NAOI shape: {true_naoi.shape}, Sample: {true_naoi[:20]}")
            true_labels = np.argmax(true_naoi, axis=1)  # Convert 2D probabilities to 1D binary labels
            print(f"Lead {lead}, Seed {seed}, Converted True Labels sample: {true_labels[:20]}, Unique values: {np.unique(true_labels, return_counts=True)}")
        else:
            true_labels = true_labels_npz  # Fallback to .npz true_labels
            print(f"Lead {lead}, Seed {seed}, Using NPZ true_labels as fallback")

        # Compute confidence scores
        confidence_scores = np.max(confidences, axis=1)
        
        # Define NAOI for both phases
        I_p_pos = confidences[:, 1]  # Predicted NAOI for positive phase (class 1)
        I_o_pos = true_labels.astype(float)  # Observed NAOI for positive phase (1 if class 1, 0 if class 0)
        I_p_neg = confidences[:, 0]  # Predicted NAOI for negative phase (class 0)
        I_o_neg = 1 - true_labels.astype(float)  # Observed NAOI for negative phase (1 if class 0, 0 if class 1)
        print(f"Lead {lead}, Seed {seed}, I_o_pos sample: {I_o_pos[:20]}, I_o_neg sample: {I_o_neg[:20]}")  # Debug

        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(I_o_pos, I_p_pos))  # Using positive phase for RMSE consistency
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, RMSE: {rmse:.4f}")
        
        # Compute correlations
        correlation_pos = np.corrcoef(I_p_pos, I_o_pos)[0, 1] if np.std(I_p_pos) > 0 and np.std(I_o_pos) > 0 else np.nan
        correlation_neg = np.corrcoef(I_p_neg, I_o_neg)[0, 1] if np.std(I_p_neg) > 0 and np.std(I_o_neg) > 0 else np.nan
        # Compute overall correlation between max confidence and true labels
        overall_correlation = np.corrcoef(confidence_scores, true_labels)[0, 1] if np.std(confidence_scores) > 0 and np.std(true_labels) > 0 else np.nan
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, NAO Correlation (Positive): {correlation_pos:.4f}")
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, NAO Correlation (Negative): {correlation_neg:.4f}")
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Overall Correlation: {overall_correlation:.4f}")
        
        # Compute weighted average as final correlation
        prop_pos = np.mean(I_o_pos)
        prop_neg = 1 - prop_pos
        final_correlation = prop_pos * correlation_pos + prop_neg * correlation_neg if not (np.isnan(correlation_pos) or np.isnan(correlation_neg)) else np.nan
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Final NAO Correlation: {final_correlation:.4f}")
        
        # Create DataFrame for this model
        df = pd.DataFrame({
            'true_label': true_labels,
            'predicted_label': predictions,
            'confidence_score': confidence_scores,
            'confidence_class_0': confidences[:, 0],
            'confidence_class_1': confidences[:, 1]
        })
        
        # Sort by confidence score
        df_sorted = df.sort_values(by='confidence_score', ascending=True)
        
        # Save sorted predictions
        csv_file = os.path.join(analysis_output_dir, f'sorted_predictions_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.csv')
        df_sorted.to_csv(csv_file, index=False)
        print(f"Saved sorted predictions to {csv_file}")
        
        # Compute accuracy by confidence bins (10 bins of 10% each)
        n_samples = len(df)
        bin_size = n_samples // 10
        bins = [
            df_sorted.iloc[i * bin_size:(i + 1) * bin_size] for i in range(10)
        ]
        if n_samples % 10 != 0:
            bins[-1] = df_sorted.iloc[9 * bin_size:]
        
        bin_accuracies = []
        for i, bin_df in enumerate(bins):
            bin_accuracy = accuracy_score(bin_df['true_label'], bin_df['predicted_label'])
            bin_accuracies.append(bin_accuracy)
            print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Bin {i+1} (Confidence {bin_df['confidence_score'].min():.4f} to {bin_df['confidence_score'].max():.4f}): Accuracy = {bin_accuracy:.4f}")
        
        overall_accuracy = accuracy_score(df['true_label'], df['predicted_label'])
        print(f"Lead {lead}, Seed {seed}, Experiment {experiment}, Overall Accuracy: {overall_accuracy:.4f}")
        
        # Append to accuracy DataFrame
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
            'overall_accuracy': overall_accuracy,
            'rmse': rmse,
            'correlation_pos': correlation_pos,
            'correlation_neg': correlation_neg,
            'final_correlation': final_correlation,
            'overall_correlation': overall_correlation
        }])], ignore_index=True)
        
        # Plot histogram of confidence scores
        plt.figure(figsize=(8, 6))
        plt.hist(confidence_scores, bins=20, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title(f'Confidence Distribution: Lead {lead}, Seed {seed}, Experiment {experiment}')
        plt.savefig(os.path.join(analysis_output_dir, f'confidence_histogram_lead_{lead}_seed_{seed}_{experiment.replace("/", "_")}_new.png'))
        plt.close()

# Save updated accuracy DataFrame
accuracy_csv = os.path.join(analysis_output_dir, 'accuracy_by_confidence_bins_with_metrics.csv')
accuracy_df.to_csv(accuracy_csv, index=False)
print(f"Saved accuracy, RMSE, and correlations by confidence bins to {accuracy_csv}")

# Plot accuracy by confidence bins
plt.figure(figsize=(12, 6))
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

# Plot NAO correlation by lead time for positive phase
plt.figure(figsize=(10, 6))
for seed in seeds:
    seed_df = accuracy_df[accuracy_df['seed'] == seed]
    plt.plot(seed_df['lead'], seed_df['correlation_pos'], marker='o', label=f'Seed {seed} (Positive)')
plt.xlabel('Lead Time (days)')
plt.ylabel('NAO Correlation Coefficient (Positive)')
plt.title('NAO Forecast Skill (Correlation) by Lead Time - Positive Phase')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(analysis_output_dir, 'nao_correlation_by_lead_time_pos.png'))
plt.close()

# Plot NAO correlation by lead time for negative phase
plt.figure(figsize=(10, 6))
for seed in seeds:
    seed_df = accuracy_df[accuracy_df['seed'] == seed]
    plt.plot(seed_df['lead'], seed_df['correlation_neg'], marker='o', label=f'Seed {seed} (Negative)')
plt.xlabel('Lead Time (days)')
plt.ylabel('NAO Correlation Coefficient (Negative)')
plt.title('NAO Forecast Skill (Correlation) by Lead Time - Negative Phase')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(analysis_output_dir, 'nao_correlation_by_lead_time_neg.png'))
plt.close()

# Plot RMSE by lead time
plt.figure(figsize=(10, 6))
for seed in seeds:
    seed_df = accuracy_df[accuracy_df['seed'] == seed]
    plt.plot(seed_df['lead'], seed_df['rmse'], marker='o', label=f'Seed {seed}')
plt.xlabel('Lead Time (days)')
plt.ylabel('RMSE')
plt.title('RMSE by Lead Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(analysis_output_dir, 'rmse_by_lead_time.png'))
plt.close()

# Plot final correlation by lead time (optional)
plt.figure(figsize=(10, 6))
for seed in seeds:
    seed_df = accuracy_df[accuracy_df['seed'] == seed]
    plt.plot(seed_df['lead'], seed_df['final_correlation'], marker='o', label=f'Seed {seed}')
plt.xlabel('Lead Time (days)')
plt.ylabel('Final NAO Correlation Coefficient')
plt.title('Final NAO Forecast Skill (Correlation) by Lead Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(analysis_output_dir, 'nao_correlation_by_lead_time_final.png'))
plt.close()

# Plot overall correlation by lead time
plt.figure(figsize=(10, 6))
for seed in seeds:
    seed_df = accuracy_df[accuracy_df['seed'] == seed]
    plt.plot(seed_df['lead'], seed_df['overall_correlation'], marker='o', label=f'Seed {seed}')
plt.xlabel('Lead Time (days)')
plt.ylabel('Overall NAO Correlation Coefficient')
plt.title('Overall NAO Forecast Skill (Correlation) by Lead Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(analysis_output_dir, 'nao_correlation_by_lead_time_overall.png'))
plt.close()