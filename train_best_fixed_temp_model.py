import os
import pickle

import backend
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import generate_dataset, filter_dataset_by_temp
from model import build_preference_model
from util import boltzmann_probability

#%% Set seeds
seed = 42
keras.utils.set_random_seed(seed)

#%% Configure experiment
N = 1000000
Tmin = 1e-2
Tmax = 1e2
candidateT = 1e-1
model_str = 'binary'
models_dir = f'models/{model_str}/n_{N}/'
for subdir in ['prefs', 'utils']:
    os.makedirs(models_dir+subdir, exist_ok=True)

FULL_MODEL = 0
UTILS_MODEL = 1

#%% Generate datasets
train, test = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmax)

#%%
fig, ax = plt.subplots()
datasets = [train]
titles = ['default']
for data, title in zip(datasets, titles):
    sns.histplot(data.T, bins=50, ax=ax, log_scale=True)
    ax.set_xlabel('T')
    ax.set_ylabel('Count')
    ax.set_title(title)
plt.tight_layout()
plt.show()
data.T.mean()

#%% Build the models
candidateTemps = np.logspace(-2, 2, 17)

name = 'best_fixed_temp'
try:
    # Load the best model
    prefs_model = keras.saving.load_model(models_dir+f'prefs/{name}.keras')
    utils_model = keras.saving.load_model(models_dir+f'utils/{name}.keras')
    model = (prefs_model, utils_model)
except ValueError:
    model_list = [build_preference_model(seed) for _ in candidateTemps]
    histories = []
    # Train the models
    for model, candidateT in zip(model_list, candidateTemps):
        data = train
        # Generate "estimated" temperatures
        temps = np.ones_like(train.T) * candidateT
        test_temps = np.ones_like(test.T) * candidateT
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="acc")]
        match model_str:
            case 'p_boltz':
                labels = data.p_boltz
            case 'binary':
                labels = data.y_boltz_2ary
            case 'ternary':
                labels = data.y_boltz_3ary
        model[FULL_MODEL].compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            metrics=metrics,
        )
        history = model[FULL_MODEL].fit(
            x=[data.x1, data.x2, temps],
            y=labels,
            batch_size=32,
            epochs=1,
            validation_data=[(test.x1, test.x2, test_temps), test.y_boltz_2ary],
        )
        history.history['T'] = candidateT
        histories.append(history.history)

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

    models = {name: model}

#%% Evaluate model probabilities
def plot_model_probs(data, model, model_name, ax=None, type='hist'):
    pref_classes = {
        '>': data.y == 1,
        '~': data.y == 0.5,
        '<': data.y == 0,
    }
    fixed_T = candidateT * np.ones(len(data.x1))
    p_hat = model[0].predict((data.x1, data.x2, fixed_T), batch_size=32)

    bins = np.linspace(0, 1, 30)

    p_hat_by_type = {
        key: p_hat[idx].squeeze() for key, idx in pref_classes.items()
    }

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Probability')
    if type == 'hist':
        sns.histplot(p_hat_by_type, alpha=0.7, palette=['C0', 'C8', 'C3'], bins=bins, kde=False, stat="count", ax=ax, legend=True)
        ax.set_ylabel('Count')
        ax.get_legend().set_loc('upper center' if 'rational' not in model_name else 'best')
    elif type == 'kde':
        sns.kdeplot(p_hat_by_type, alpha=0.7, palette=['C0', 'C8', 'C3'], ax=ax, common_norm=False, clip=[0,1])
        ax.set_ylabel('Density')
    else:
        raise ValueError(f"Unknown 'type' parameter: {type}")
    if should_show_plot:
        ax.show()

plot_models = [
    # 'skyline',
    # 'rational',
    # 'true_temp',
    # 'gt_utils',
    'best_fixed_temp',
]
for type in ['hist', 'kde']:
    fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True, sharey=(type=='hist'))
    for name, ax in zip(plot_models, axes.flatten()):
        plot_model_probs(test, models[name], name, ax=ax, type=type)
    plt.tight_layout()
    plt.show()


#%% Evaluate model utilities
def plot_utility_calibration(data, model, model_name, ax=None):
    fixed_T = candidateT * np.ones(len(data.x1))
    u1_hat, _ = model[1].predict((data.x1, data.x2, fixed_T), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Actual utility')
    sns.violinplot(x=9*test.u1, y=9*u1_hat.squeeze(), ax=ax, legend=True)
    ax.set_ylim([0,9])
    ax.set_ylabel('Predicted utility')
    if should_show_plot:
        ax.show()


fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True)
for name, ax in zip(plot_models, axes.flatten()):
    plot_utility_calibration(test, models[name], name, ax=ax)
plt.tight_layout()
plt.show()

#%%

def plot_probability_calibration(model, model_name, ax=None):
    p_hat = model[0].predict((test.x1, test.x2, test.T), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel(r'Actual $p_{boltz}$')
    sns.kdeplot(x=test.p_boltz_2ary, y=p_hat.squeeze(), clip=[0,1], ax=ax, legend=True)
    ax.set_ylim([0,1])
    ax.set_ylabel(r'Predicted $\hat p_{boltz}$')
    if should_show_plot:
        ax.show()


fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True, sharey=True)
for name, ax in zip(plot_models, axes.flatten()):
    plot_probability_calibration(models[name], name, ax=ax)
plt.tight_layout()
plt.show()
