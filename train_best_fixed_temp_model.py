import os
import pickle

import backend
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import generate_dataset, filter_dataset_by_temp
from model import build_preference_model, load_model, FULL_MODEL, UTILS_MODEL
from util import boltzmann_probability
from plotting import plot_temp_histogram, plot_model_probs, plot_utility_calibration, plot_probability_calibration

#%% Set seeds
seed = 42
keras.utils.set_random_seed(seed)

#%% Configure experiment
N = 1000000
Tmin = 1e-2
Tmax = 1e2
eval_temp = 1e-2
model_str = 'binary'
models_dir = f'models/{model_str}/n_{N}/'
for subdir in ['prefs', 'utils']:
    os.makedirs(models_dir+subdir, exist_ok=True)

#%% Generate datasets
train, test = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmax)
plot_temp_histogram(train)

#%%

def train_candidate_model(model, train_data, val_data, candidateT):
    # Generate "estimated" temperatures
    temps = np.ones_like(train_data.T) * candidateT
    val_temps = np.ones_like(val_data.T) * candidateT

    loss = keras.losses.BinaryCrossentropy()
    metrics = [keras.metrics.BinaryAccuracy(name="acc")]

    # Grab the appropriate labels
    match model_str:
        case 'p_boltz':
            labels = train_data.p_boltz
        case 'binary':
            labels = train_data.y_boltz_2ary
        case 'ternary':
            labels = train_data.y_boltz_3ary

    # Train
    model[FULL_MODEL].compile(
        loss=loss,
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=metrics,
    )
    history = model[FULL_MODEL].fit(
        x=[train_data.x1, train_data.x2, temps],
        y=labels,
        batch_size=32,
        epochs=1,
        validation_data=[(val_data.x1, val_data.x2, val_temps), val_data.y_boltz_2ary],
    )
    history.history['T'] = candidateT
    return history.history

#%% Load best model or run a new sweep
name = 'best_fixed_temp'
try:
    model = load_model(models_dir, name)
except ValueError:
    # Build the models
    candidateTemps = np.logspace(-2, 2, 17)
    model_list = [build_preference_model(seed) for _ in candidateTemps]
    histories = []
    # Train the models
    for model, candidateT in zip(model_list, candidateTemps):
        history = train_candidate_model(model, train, test, candidateT)
        histories.append(history)

    # Plot accuracies vs. temperatures
    accuracies = [history['val_acc'][0] for history in histories]
    fig, ax = plt.subplots()
    ax.plot(candidateTemps, accuracies)
    ax.semilogx()
    ax.set_xlabel('T')
    ax.set_ylabel('Test Accuracy')
    plt.show()

    # Save the best model
    acc, best_fixed_T, model = max(zip(accuracies, candidateTemps, model_list), key=lambda x: x[0])
    keras.saving.save_model(model[FULL_MODEL], models_dir+f'prefs/{name}.keras')
    keras.saving.save_model(model[UTILS_MODEL], models_dir+f'utils/{name}.keras')
    best_fixed_temps = [
        {'n': 100000, 'T': 0.03162277660168379},
        {'n': 1000000, 'T': 0.1778279410038923},
    ]
