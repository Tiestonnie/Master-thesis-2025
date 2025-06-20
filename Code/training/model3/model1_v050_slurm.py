import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers, metrics
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import accuracy_score
import settings_model1_v050 as settings
import random
import os
import sys

# Check for GPU availability
print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# Constants
NLABEL = 2  # Binary classification for NAO

# Directories
ddir_X = "/gpfs/home4/tleneman/Data/Processed_cesm2_combined/"  # Combined SLP + v050 data
ddir_Y = "/gpfs/home4/tleneman/Data/Processed_cesm2/"  # Labels
ddir_out = "/gpfs/home4/tleneman/model1_v050/"
os.makedirs(ddir_out, exist_ok=True)

# Function to load data with error handling and NaN/infinite value imputation
def load_data(file_path, subset_size=None):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    data = np.load(file_path)
    if subset_size is not None:
        data = data[:subset_size]
    # Check for NaNs and infinite values
    nan_mask = np.isnan(data)
    nan_count = np.sum(nan_mask)
    inf_count = np.isinf(data).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {nan_count} NaNs and {inf_count} infinite values detected in {file_path}!")
        if data.ndim == 2:  # For X data
            column_means = np.nanmean(data, axis=0)
            data = np.where(nan_mask, np.tile(column_means, (data.shape[0], 1)), data)
        else:  # For Y data
            raise ValueError("NaNs or infinite values found in labels. Please fix preprocessing.")
    return data

# Function to ensure input is a list
def ensure_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]

# Function for days_average (for compatibility, though not used here)
def apply_averaging(data, days):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    kernel = np.ones(days) / days
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), 1, data)
    return smoothed

# Learning rate scheduler
def scheduler(epoch, lr):
    if epoch < 10:
        return float(lr)
    return float(lr * np.exp(-0.1))

# Model definition
def defineNN(hidden, input_shape, output_shape, ridge_penalty1=0., lasso_penalty1=0., dropout=0., act_fun='relu', network_seed=99):
    tf.keras.backend.clear_session()
    tf.random.set_seed(network_seed)
    
    # Input layer
    input_layer = tf.keras.Input(shape=input_shape)
    
    # Optional dropout
    if dropout > 0.:
        x = tf.keras.layers.Dropout(rate=dropout, seed=network_seed)(input_layer)
    else:
        x = input_layer
    
    # Dense layers
    x = tf.keras.layers.Dense(
        hidden[0],
        activation=act_fun,
        use_bias=True,
        kernel_regularizer=regularizers.l1_l2(l1=lasso_penalty1, l2=ridge_penalty1),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
    )(x)
    
    for layer in hidden[1:]:
        x = tf.keras.layers.Dense(
            layer,
            activation=act_fun,
            use_bias=True,
            kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=0.0),
            bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
            kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
        )(x)
    
    output_layer = tf.keras.layers.Dense(
        output_shape,
        activation=tf.keras.activations.softmax,
        use_bias=True,
        kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=0.0),
        bias_initializer=tf.keras.initializers.RandomNormal(seed=network_seed),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=network_seed)
    )(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    
    return model

class PredictionAccuracy(metrics.Metric):
    def __init__(self, nlabel, name='prediction_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.nlabel = nlabel
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.float32))
        total = tf.cast(tf.shape(y_true)[0], tf.float32)
        self.correct.assign_add(correct)
        self.total.assign_add(total)
    
    def result(self):
        return self.correct / self.total
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

def make_model(X_train, NLABEL, HIDDENS, RIDGE1, DROPOUT, LR_INIT, NETWORK_SEED):
    tf.keras.backend.clear_session()
    model = defineNN(
        hidden=HIDDENS,
        input_shape=X_train.shape[1:],
        output_shape=NLABEL,
        ridge_penalty1=RIDGE1,
        lasso_penalty1=0.0,
        dropout=DROPOUT,
        act_fun='relu',
        network_seed=NETWORK_SEED
    )
    
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_INIT),
        loss=loss_function,
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            PredictionAccuracy(NLABEL)
        ]
    )
    
    return model, loss_function

# Get experiment from command-line argument
#if len(sys.argv) != 2:
#    raise ValueError("Please provide an experiment name as a command-line argument")
#EXPERIMENT = sys.argv[1]

if len(sys.argv) != 3:
    raise ValueError("Please provide experiment name and seed string")
EXPERIMENT = sys.argv[1]
random_seeds = [int(s) for s in sys.argv[2].split()]


print(f"\nRunning experiment: {EXPERIMENT}")
params = settings.get_settings(EXPERIMENT)

# Use lead times from settings
lead_times = [params['lead']]
print(f"Lead times for this experiment: {lead_times}")

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
    # Load training data
    X_train_list = []
    Y_train_dict = {lead: [] for lead in lead_times}
    for ens in train_ens:
        for lead in lead_times:
            X_file = os.path.join(ddir_X, f'member_{ens}_lead_{lead}.npy')  # Combined SLP + v050
            X = load_data(X_file)
            print(f"Shape of X for ensemble {ens}, lead {lead}: {X.shape}")
            X_train_list.append(X)

            labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
            labels_dict = np.load(labels_file)
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict[lead_key]
            print(f"Shape of Y for ensemble {ens}, lead {lead}: {Y.shape}")
            Y_train_dict[lead].append(Y)

    X_train = np.concatenate(X_train_list, axis=0)
    for lead in lead_times:
        Y_train_dict[lead] = np.concatenate(Y_train_dict[lead], axis=0)
        print(f"Shape of concatenated X_train: {X_train.shape}")
        print(f"Shape of concatenated Y_train for lead {lead}: {Y_train_dict[lead].shape}")
        if X_train.shape[0] != Y_train_dict[lead].shape[0]:
            raise ValueError(f"Mismatch in sample sizes: X_train has {X_train.shape[0]}, Y_train has {Y_train_dict[lead].shape[0]}")

    # Load validation data
    X_val_list = []
    Y_val_dict = {lead: [] for lead in lead_times}
    for ens in val_ens:
        for lead in lead_times:
            X_file = os.path.join(ddir_X, f'member_{ens}_lead_{lead}.npy')
            X = load_data(X_file)
            X_val_list.append(X)

            labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
            labels_dict = np.load(labels_file)
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict[lead_key]
            Y_val_dict[lead].append(Y)

    X_val = np.concatenate(X_val_list, axis=0)
    for lead in lead_times:
        Y_val_dict[lead] = np.concatenate(Y_val_dict[lead], axis=0)

    # Load test data
    X_test_list = []
    Y_test_dict = {}
    for ens in test_ens:
        for lead in lead_times:
            X_file = os.path.join(ddir_X, f'member_{ens}_lead_{lead}.npy')
            X = load_data(X_file)
            X_test_list.append(X)

            labels_file = os.path.join(ddir_Y, f'labels_{ens}.npz')
            labels_dict = np.load(labels_file)
            lead_key = f'lead_{lead}'
            if lead_key not in labels_dict:
                raise KeyError(f"Key {lead_key} not found in {labels_file}")
            Y = labels_dict[lead_key]
            Y_test_dict[lead] = Y

    X_test = np.concatenate(X_test_list, axis=0)

    # Train networks per lead time with different seeds
    for seed in random_seeds:
        for lead in lead_times:
            print(f"\nTraining model for lead time {lead} days, seed {seed}, iteration {iteration + 1}")

            exp_settings = settings.get_settings(EXPERIMENT)
            # Apply days_average if needed (currently not used, but kept for compatibility)
            Y_train = Y_train_dict[lead]
            Y_val = Y_val_dict[lead]

            print("Y_train shape:", Y_train.shape)
            print("Number of classes (NLABEL):", Y_train.shape[1])
            model, loss_function = make_model(
                X_train=X_train,
                NLABEL=NLABEL,
                HIDDENS=exp_settings['HIDDENS'],
                RIDGE1=exp_settings['RIDGE1'],
                DROPOUT=exp_settings['DROPOUT'],
                LR_INIT=exp_settings['LR_INIT'],
                NETWORK_SEED=seed
            )
            
            history = model.fit(
                X_train, Y_train,
                epochs=exp_settings['N_EPOCHS'],
                batch_size=exp_settings['BATCH_SIZE'],
                validation_data=(X_val, Y_val),
                callbacks=[
                    EarlyStopping(patience=exp_settings['PATIENCE'], restore_best_weights=True),
                    LearningRateScheduler(scheduler)
                ],
                verbose=1,
                class_weight=exp_settings.get('CLASS_WEIGHT', None)
            )

            # Predict on test set
            Y_pred_prob = model.predict(X_test)
            Y_pred = np.argmax(Y_pred_prob, axis=1)

            # Compute accuracy
            Y_test = Y_test_dict[lead]
            test_accuracy = accuracy_score(np.argmax(Y_test, axis=1), Y_pred)
            print(f"Lead {lead} days - Test Accuracy: {test_accuracy:.4f}")

            # Store results
            if lead not in results:
                results[lead] = {'accuracies': [], 'Y_preds': [], 'Y_tests': []}
            results[lead]['accuracies'].append(test_accuracy)
            results[lead]['Y_preds'].append(Y_pred)
            results[lead]['Y_tests'].append(np.argmax(Y_test, axis=1))

            # Save model
            model_file = os.path.join(ddir_out, f'model_lead_{lead}_seed_{seed}_{EXPERIMENT.replace("/", "_")}.h5')
            model.save(model_file)

            # Check label distribution
            print(f"Lead {lead} days - Training label distribution:", np.bincount(np.argmax(Y_train, axis=1)))
            print(f"Lead {lead} days - Validation label distribution:", np.bincount(np.argmax(Y_val, axis=1)))
            print(f"Lead {lead} days - Test label distribution:", np.bincount(np.argmax(Y_test, axis=1)))

# Compute average accuracy per lead time
for lead in lead_times:
    avg_accuracy = np.mean(results[lead]['accuracies'])
    print(f"\nAverage Test Accuracy for lead {lead} days across all iterations and seeds in {EXPERIMENT}: {avg_accuracy:.4f}")