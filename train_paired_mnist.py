import os
import pickle

import backend
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import generate_dataset, filter_dataset_by_temp
from model import build_preference_model, load_model
from util import boltzmann_probability
from plotting import plot_temp_histograms, plot_model_probs, plot_utility_calibration, plot_probability_calibration

#%% Set seeds
seed = 42
keras.utils.set_random_seed(seed)

#%% Configure experiment
N = 1000000
Tmin = 1e-2
Tmax = 1e2
model_str = 'binary'
models_dir = f'models/{model_str}/n_{N}/'
for subdir in ['prefs', 'utils']:
    os.makedirs(models_dir+subdir, exist_ok=True)

#%% Generate datasets
train, test = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmax)

train_tmin, test_tmin = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmin)

percentile = 0.1
temp_cutoff = sorted(train.T)[int(N*percentile)]
train_low_temp = filter_dataset_by_temp(train, tmax=temp_cutoff)

#%% Generate "estimated" temperatures
fixed_Tmin = np.ones_like(train.T) * Tmin

#%%
datasets = [train, train_tmin, train_low_temp]
titles = ['default', 'tmin', 'low_temp']
plot_temp_histograms(datasets, titles)

#%% Build the models
model_settings = {
    #name: (training_subcomponent, )
    'true_temp': (train, None),
    'rational': (train, fixed_Tmin),
    'skyline': (train_tmin, None),
    'low_temp': (train_low_temp, None),
    'gt_utils': (train, None),
}
models = {name: build_preference_model(seed) for name in model_settings.keys()}

try:
    # Load the models
    for name, model_tuple in models.items():
        models[name] = load_model(models_dir, name)
except ValueError:
    # Train the models
    for name, settings in model_settings.items():
        data, temps = settings
        if temps is None:
            temps = data.T
        if name != 'gt_utils':
            subcomponent = 0
            loss = keras.losses.BinaryCrossentropy()
            metrics = [keras.metrics.BinaryAccuracy(name="acc")]
            match model_str:
                case 'p_boltz':
                    labels = data.p_boltz
                case 'binary':
                    labels = data.y_boltz_2ary
                case 'ternary':
                    labels = data.y_boltz_3ary
        else:
            subcomponent = 1
            loss=[keras.losses.MeanSquaredError(), keras.losses.MeanSquaredError()],
            metrics=[
                keras.metrics.MeanSquaredError(name="mse1"),
                keras.metrics.MeanSquaredError(name="mse2"),
            ]
            labels=[data.u1, data.u2]
        models[name][subcomponent].compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=metrics,
        )
        models[name][subcomponent].fit(
            x=[data.x1, data.x2, temps],
            y=labels,
            batch_size=32,
            epochs=1,
        )
    # Save the models
    for name, (pref_model, utils_model) in models.items():
        keras.saving.save_model(pref_model, models_dir+f'prefs/{name}.keras')
        keras.saving.save_model(utils_model, models_dir+f'utils/{name}.keras')
