import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import settings_model2 as settings
import random
import os

# Controleer of GPU beschikbaar is
print("Beschikbare GPU's:", tf.config.list_physical_devices('GPU'))

# experiments = ['exp_2/exp_200', 'exp_3/_exp300', 'exp_4/exp_400', 'exp_5/exp_500', 'exp_6/exp_600',
#               'exp_7/exp_700', 'exp_8/exp_800']

# Alle experimenten
# experiments = ['exp_2/exp_200', 'exp_2/exp_201', 'exp_2/exp_202', 'exp_2/exp_203', 'exp_2/exp_204',
#               'exp_3/exp_300', 'exp_3/exp_301', 'exp_3/exp_302', 'exp_3/exp_303', 'exp_3/exp_304',
#               'exp_4/exp_400', 'exp_4/exp_401', 'exp_4/exp_402', 'exp_4/exp_403', 'exp_4/exp_404',
#               'exp_5/exp_500', 'exp_5/exp_501', 'exp_5/exp_502', 'exp_5/exp_503', 'exp_5/exp_504',
#               'exp_6/exp_600', 'exp_6/exp_601', 'exp_6/exp_602', 'exp_6/exp_603', 'exp_6/exp_604',
#               'exp_7/exp_700', 'exp_7/exp_701', 'exp_7/exp_702', 'exp_7/exp_703', 'exp_7/exp_704',
#               'exp_8/exp_800', 'exp_8/exp_801', 'exp_8/exp_802', 'exp_8/exp_803', 'exp_8/exp_804']


# Local directories (adjust paths to your data)
ddir_X = "/gpfs/home4/tleneman/Data/Processed_cesm2_3day/"
ddir_Y = "/gpfs/home4/tleneman/Data/Processed_cesm2_3day/"
ddir_out = "/gpfs/home4/tleneman/model2_3day/"

# Ensure output directory exists
os.makedirs(ddir_out, exist_ok=True)

# Function to load data with error handling
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    return np.load(file_path)

# Function to ensure input is a list
def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]

import sys

# Get experiment from command-line argument
if len(sys.argv) != 2:
    raise ValueError("Please provide an experiment name as a command-line argument")
EXPERIMENT = sys.argv[1]

print(f"\nRunning experiment: {EXPERIMENT}")
params = settings.get_settings(EXPERIMENT)
# Rest of the code (from lead_times = [params['lead']] onward) remains the same



# Use the single lead time from settings
lead_times = [params['lead']]

# Full run settings
n_iterations = 1
random_seeds = params['RANDOM_SEED']

# Use ensemble members from settings
train_ens = [str(ens) for ens in ensure_list(params['training_ens'])]
val_ens = [str(ens) for ens in ensure_list(params['validation_ens'])]
test_ens = [str(ens) for ens in ensure_list(params['testing_ens'])]

# Run iterations
results = {}
for iteration in range(n_iterations):
    # Load full training data
    X_train_list = []
    Y_train_dict = {lead: [] for lead in lead_times}
    for ens in train_ens:
        X_file = ddir_X + f'3day_member_{ens}_lead_{lead_times[0]}.npy'
        X = load_data(X_file)
        print(f"Shape of X for ensemble {ens}, lead {lead_times[0]}: {X.shape}")
        X_train_list.append(X)
            
        labels_file = ddir_Y + f'labels_{ens}.npz'
        labels_dict = load_data(labels_file)
        for lead in lead_times:
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict[lead_key]
            Y = np.argmax(Y, axis=1)
            print(f"Shape of Y for ensemble {ens}, lead {lead}: {Y.shape}")
            Y_train_dict[lead].append(Y)

    X_train = np.concatenate(X_train_list, axis=0)
    for lead in lead_times:
        Y_train_dict[lead] = np.concatenate(Y_train_dict[lead], axis=0)
        print(f"Shape of concatenated X_train: {X_train.shape}")
        print(f"Shape of concatenated Y_train for lead {lead}: {Y_train_dict[lead].shape}")
        if X_train.shape[0] != Y_train_dict[lead].shape[0]:
            raise ValueError(f"Mismatch in sample sizes: X_train has {X_train.shape[0]}, Y_train has {Y_train_dict[lead].shape[0]}")

    # Load full validation data
    X_val_list = []
    Y_val_dict = {lead: [] for lead in lead_times}
    for ens in val_ens:
        X_file = ddir_X + f'member_{ens}.npy'
        X = load_data(X_file)
        X_val_list.append(X)
          
        labels_file = ddir_Y + f'labels_{ens}.npz'
        labels_dict = load_data(labels_file)
        for lead in lead_times:
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict[lead_key]
            Y = np.argmax(Y, axis=1)
            Y_val_dict[lead].append(Y)

    X_val = np.concatenate(X_val_list, axis=0)
    for lead in lead_times:
        Y_val_dict[lead] = np.concatenate(Y_val_dict[lead], axis=0)

    # Load full testing data
    X_test_list = []
    Y_test_dict = {}
    for ens in test_ens:
        X_file = ddir_X + f'member_{ens}.npy'
        X = load_data(X_file)
        X_test_list.append(X)
            
        labels_file = ddir_Y + f'labels_{ens}.npz'
        labels_dict_test = load_data(labels_file)
        for lead in lead_times:
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict_test:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict_test[lead_key]
            Y_test_dict[lead] = np.argmax(Y, axis=1)

    X_test = np.concatenate(X_test_list, axis=0)

    # Define model dynamically based on HIDDENS
    def make_model(input_shape, network_seed, output_shape=1, output_activation='sigmoid', ridge_penalty1=0., lasso_penalty1=0., dropout=0.):
        tf.random.set_seed(network_seed)
        input_layer = Input(shape=(input_shape,))
        if dropout > 0.:
            x = Dropout(rate=dropout, seed=network_seed)(input_layer)
            x = Dense(params['HIDDENS'][0], activation=params['act_fun'],
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lasso_penalty1, l2=ridge_penalty1),
                      bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(x)
        else:
            x = Dense(params['HIDDENS'][0], activation=params['act_fun'],
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=lasso_penalty1, l2=ridge_penalty1),
                      bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(input_layer)

        for units in params['HIDDENS'][1:]:
            x = Dense(units, activation=params['act_fun'],
                      kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0),
                      bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(x)

        output_layer = Dense(output_shape, activation=output_activation,
                             kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0),
                             bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
                             kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed))(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['LR_INIT']),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        return model

    # Train networks per lead time with different seeds
    for seed in random_seeds:
        for lead in lead_times:
            print(f"\nTraining model for lead time {lead} days, seed {seed}, iteration {iteration + 1}")
               
            model = make_model(input_shape=X_train.shape[1], network_seed=seed, output_shape=1, output_activation='sigmoid',
                               ridge_penalty1=params['RIDGE1'], lasso_penalty1=0., dropout=params['DROPOUT'])
            history = model.fit(X_train, Y_train_dict[lead],
                                epochs=params['N_EPOCHS'],
                                batch_size=params['BATCH_SIZE'],
                                validation_data=(X_val, Y_val_dict[lead]),
                                callbacks=[EarlyStopping(patience=params['PATIENCE'])],
                                class_weight=params.get('CLASS_WEIGHT', None),
                                verbose=1)

            # Predict on test set
            Y_pred_prob = model.predict(X_test)
            Y_pred = (Y_pred_prob > 0.5).astype(int).flatten()

            # Compute accuracy
            test_accuracy = accuracy_score(Y_test_dict[lead], Y_pred)
            print(f"Lead {lead} days - Test Accuracy: {test_accuracy:.4f}")

            # Store results
            if lead not in results:
                results[lead] = {'accuracies': [], 'Y_preds': [], 'Y_tests': []}
            results[lead]['accuracies'].append(test_accuracy)
            results[lead]['Y_preds'].append(Y_pred)
            results[lead]['Y_tests'].append(Y_test_dict[lead])

            # Save model locally
            model_file = ddir_out + f'model_3day_lead_{lead}_seed_{seed}_{EXPERIMENT.replace("/", "_")}.h5'
            model.save(model_file)

            # Check label distribution
            print(f"Lead {lead} days - Training label distribution:", np.bincount(Y_train_dict[lead]))
            print(f"Lead {lead} days - Validation label distribution:", np.bincount(Y_val_dict[lead]))
            print(f"Lead {lead} days - Test label distribution:", np.bincount(Y_test_dict[lead]))

# Compute average accuracy per lead time
for lead in lead_times:
    avg_accuracy = np.mean(results[lead]['accuracies'])
    print(f"\nAverage Test Accuracy for lead {lead} days across all iterations and seeds in {EXPERIMENT}: {avg_accuracy:.4f}")