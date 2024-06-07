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

#%%
plot_models = [
    # 'skyline',
    ('rational', r'$\beta_{max}$'),
    ('best_fixed_temp', r'$\beta_{best}$'),
    ('true_temp', r'$\beta$'),
    # 'low_temp',
    ('gt_utils', r'$U(x)$'),
]
models = {spec[0]: load_model(models_dir, spec[0]) for spec in plot_models}

#%% Evaluate model probabilities
for type in ['hist', 'kde']:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=(type=='hist'))
    for (name, title), ax in zip(plot_models, axes.flatten()):
        plot_model_probs(test, models[name], title, ax=ax, type=type)
    plt.tight_layout()
    plt.show()

#%% Evaluate model utilities
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
for (name, title), ax in zip(plot_models, axes.flatten()):
    plot_utility_calibration(test, models[name], title, ax=ax)
plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
for (name, title), ax in zip(plot_models, axes.flatten()):
    plot_probability_calibration(test, models[name], title, ax=ax)
plt.tight_layout()
plt.show()
