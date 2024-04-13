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

#%% Generate datasets
train, test = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmax)

train_tmin, test_tmin = generate_dataset(n_train=N, tmin=Tmin, tmax=Tmin)

percentile = 0.1
temp_cutoff = sorted(train.T)[int(N*percentile)]
train_low_temp = filter_dataset_by_temp(train, tmax=temp_cutoff)

#%% Generate "estimated" temperatures
fixed_Tmin = np.ones_like(train.T) * Tmin

#%%
fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True, sharey=True)
datasets = [train, train_tmin, train_low_temp]
titles = ['default', 'tmin', 'low_temp']
for data, title, ax in zip(datasets, titles, axes):
    sns.histplot(data.p_boltz, bins=50, ax=ax)
    ax.set_xlabel('p_boltz')
    ax.set_ylabel('Count')
    ax.set_title(title)
plt.tight_layout()
plt.show()

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

for name, settings in model_settings.items():
    data, temps = settings
    if temps is None:
        temps = data.T
    if name != 'gt_utils':
        subcomponent = 0
        loss = keras.losses.BinaryCrossentropy()
        metrics = [keras.metrics.BinaryAccuracy(name="acc")]
        labels = data.y_hat
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

#%% Evaluate model probabilities
def plot_model_probs(data, model, model_name, ax=None, type='hist'):
    pref_classes = {
        '>': data.y == 1,
        '~': data.y == 0.5,
        '<': data.y == 0,
    }
    fixed_Tmin = Tmin * np.ones(len(data.x1))
    p_hat = model[0].predict((data.x1, data.x2, fixed_Tmin), batch_size=32)

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
    'rational',
    'true_temp',
    # 'low_temp',
    'gt_utils',
]
for type in ['hist', 'kde']:
    fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True, sharey=True)
    for name, ax in zip(plot_models, axes.flatten()):
        plot_model_probs(test, models[name], name, ax=ax, type=type)
    plt.tight_layout()
    plt.show()


#%% Evaluate model utilities
def plot_utility_calibration(data, model, model_name, ax=None):
    fixed_Tmin = Tmin * np.ones(len(data.x1))
    u1_hat, _ = model[1].predict((data.x1, data.x2, fixed_Tmin), batch_size=32)

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
    sns.kdeplot(x=test.p_boltz, y=p_hat.squeeze(), clip=[0,1], ax=ax, legend=True)
    ax.set_ylim([0,1])
    ax.set_ylabel(r'Predicted $\hat p_{boltz}$')
    if should_show_plot:
        ax.show()


fig, axes = plt.subplots(1,3, figsize=(10, 4), sharex=True, sharey=True)
for name, ax in zip(plot_models, axes.flatten()):
    plot_probability_calibration(models[name], name, ax=ax)
plt.tight_layout()
plt.show()
