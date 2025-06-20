import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
import os
import sys

# Directories
ddir_X = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined/"
ddir_Y = "/gpfs/home4/tleneman/Data/Processed_cesm2/"
ddir_out = "/gpfs/home4/tleneman/xgboost_models/"

os.makedirs(ddir_out, exist_ok=True)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    return np.load(file_path)

if len(sys.argv) != 3:
    raise ValueError("Please provide experiment name and seed string")
EXPERIMENT = sys.argv[1]
seeds = [int(s) for s in sys.argv[2].split()]
import settings_model1 as settings
params = settings.get_settings(EXPERIMENT)
lead_times = [params['lead']]
train_ens = [str(ens) for ens in params['training_ens']]
val_ens = [str(ens) for ens in params['validation_ens']] if isinstance(params['validation_ens'], (list, tuple)) else [str(params['validation_ens'])]
test_ens = [str(ens) for ens in params['testing_ens']] if isinstance(params['testing_ens'], (list, tuple)) else [str(params['testing_ens'])]

# Load and prepare data
X_train_list, Y_train_list = [], []
for ens in train_ens:
    X = load_data(os.path.join(ddir_X, f'member_{ens}_lead_{lead_times[0]}.npy'))
    labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
    Y = np.load(labels_file)[f'lead_{lead_times[0]}']
    X_train_list.append(X)
    Y_train_list.append(Y)
X_train = np.concatenate(X_train_list, axis=0)
Y_train = np.concatenate(Y_train_list, axis=0)
Y_train = np.argmax(Y_train, axis=1)
if len(np.unique(Y_train)) < 2:
    raise ValueError(f"Y_train contains only {np.unique(Y_train)} classes, expected 2 for binary classification")
print(f"Unique labels in Y_train: {np.unique(Y_train)}")

X_val_list, Y_val_list = [], []
for ens in val_ens:
    X = load_data(os.path.join(ddir_X, f'member_{ens}_lead_{lead_times[0]}.npy'))
    labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
    Y = np.load(labels_file)[f'lead_{lead_times[0]}']
    X_val_list.append(X)
    Y_val_list.append(Y)
X_val = np.concatenate(X_val_list, axis=0)
Y_val = np.concatenate(Y_val_list, axis=0)
Y_val = np.argmax(Y_val, axis=1)
if len(np.unique(Y_val)) < 2:
    raise ValueError(f"Y_val contains only {np.unique(Y_val)} classes, expected 2 for binary classification")
print(f"Unique labels in Y_val: {np.unique(Y_val)}")

X_test_list, Y_test_list = [], []
for ens in test_ens:
    X = load_data(os.path.join(ddir_X, f'member_{ens}_lead_{lead_times[0]}.npy'))
    labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
    Y = np.load(labels_file)[f'lead_{lead_times[0]}']
    X_test_list.append(X)
    Y_test_list.append(Y)
X_test = np.concatenate(X_test_list, axis=0)
Y_test = np.concatenate(Y_test_list, axis=0)
Y_test = np.argmax(Y_test, axis=1)
if len(np.unique(Y_test)) < 2:
    raise ValueError(f"Y_test contains only {np.unique(Y_test)} classes, expected 2 for binary classification")
print(f"Unique labels in Y_test: {np.unique(Y_test)}")

# Train XGBoost
for seed in seeds:
    print(f"\nTraining XGBoost for lead {lead_times[0]} days, seed {seed}")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=seed,
        use_label_encoder=False,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    eval_set = [(X_val, Y_val)]
    xgb_model.fit(X_train, Y_train, eval_set=eval_set, verbose=True)

    # Predict and evaluate
    Y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Lead {lead_times[0]} days - Test Accuracy: {accuracy:.4f}")

    # Save model
    model_file = os.path.join(ddir_out, f'xgb_model_lead_{lead_times[0]}_seed_{seed}_{EXPERIMENT.replace("/", "_")}.json')
    xgb_model.save_model(model_file)