import numpy as np
import shap
import matplotlib.pyplot as plt

# Load the SHAP values file
file_path = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/shap/shap_values_lead_35_seed_47_exp_6_exp_600.h5_new.npz"
data = np.load(file_path)
shap_values = data['shap_values']
feature_names = data['feature_names']

# Load the corresponding X_test data (assuming it's available)
x_test_path = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined/member_0_lead_35.npy"
X_test = np.load(x_test_path)

# Debug: Print shapes
print(f"SHAP values shape: {shap_values.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Feature names length: {len(feature_names)}")

# Handle the shape mismatch
if shap_values.shape[0] == len(feature_names) and shap_values.shape[1] == 2:
    print("SHAP values appear to be mean SHAP values per feature for two classes.")
    # Use mean absolute SHAP values for bar plot (no sample-wise summary plot possible)
    mean_shap = np.mean(np.abs(shap_values), axis=1)  # Average across classes
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, plot_type="bar", feature_names=feature_names, show=False)
    plt.title(f"SHAP Bar Plot - Lead 35, Seed 47, Exp exp_6_exp_600 (Mean per Feature)")
    bar_plot_file = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/shap/shap_bar_lead_35_seed_47_exp_6_exp_600.png"
    plt.savefig(bar_plot_file, bbox_inches='tight')
    plt.close()
    print(f"Bar plot saved to {bar_plot_file}")
else:
    # Assume full SHAP values (samples, features) if shape matches
    if shap_values.shape[1] == X_test.shape[1]:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - Lead 35, Seed 47, Exp exp_6_exp_600")
        summary_plot_file = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/shap/shap_summary_lead_35_seed_47_exp_6_exp_600.png"
        plt.savefig(summary_plot_file, bbox_inches='tight')
        plt.close()
        print(f"Summary plot saved to {summary_plot_file}")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        plt.title(f"SHAP Bar Plot - Lead 35, Seed 47, Exp exp_6_exp_600")
        bar_plot_file = "/gpfs/home4/tleneman/model1_v050/results_test_new/analysis/shap/shap_bar_lead_35_seed_47_exp_6_exp_600.png"
        plt.savefig(bar_plot_file, bbox_inches='tight')
        plt.close()
        print(f"Bar plot saved to {bar_plot_file}")
    else:
        raise ValueError(f"Unexpected SHAP values shape {shap_values.shape} - cannot visualize with X_test shape {X_test.shape}")
