import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Set seeds
seed = 42
keras.utils.set_random_seed(seed)

def generate_pref_pairs_dataset(x, y, N=None):
    # Generate (input, input) -> (output, output) examples from MNIST (x, y) data

    x = x.astype("float32") / 255 # Scale images to the [0, 1] range
    y = y.astype("float32") / 9 # Scale labels to the [0, 1] range

    # Make sure images have shape (28, 28, 1)
    x = np.expand_dims(x, -1)

    # The (x1, x2) values consitute a comparison between two inputs, and the y values
    # are the corresponding ground truth utilities for the individual inputs.
    # Preferences are assumed to be Boltzmann distributed relative to these utilities,
    # with varying temperatures.

    # The goal is to train a model to predict the utilities from the inputs and preferences

    # Generate random indexes for the first and second inputs
    if N is not None:
        idx1 = np.random.randint(0, len(x), size=(N,))
        idx2 = np.random.randint(0, len(x), size=(N,))
    else:
        idx1 = np.arange(len(x))
        idx2 = np.arange(len(x))
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)

    # Select input pairs and corresponding 'utilities'
    x1 = x[idx1]
    x2 = x[idx2]
    u1 = y[idx1]
    u2 = y[idx2]
    return (x1, x2), (u1, u2)

def boltzmann_probability(utility1, utility2, temp):
    max_utility = np.maximum(utility1, utility2)
    # Subtract the max utility to prevent overflow
    exp_diff1 = np.exp((utility1 - max_utility) / temp)
    exp_diff2 = np.exp((utility2 - max_utility) / temp)
    return exp_diff1 / (exp_diff1 + exp_diff2)

#%%
# Import MNIST
N = 1000000
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
(x1_train, x2_train), (u1_train, u2_train) = generate_pref_pairs_dataset(x_train, y_train, N)
(x1_test, x2_test), (u1_test, u2_test) = generate_pref_pairs_dataset(x_test, y_test)

# Generate temps and sample preferences from Boltzmann distribution
temps = 10**np.random.uniform(low=-2., high=2., size=N).astype(np.float32)
Tmin = min(temps)
pr_prefer_1st = boltzmann_probability(u1_train, u2_train, temps)
pr_prefer_1st_tmin = boltzmann_probability(u1_train, u2_train, Tmin*np.ones_like(temps))
prefs = np.random.binomial(1, pr_prefer_1st)
prefs_tmin = np.random.binomial(1, pr_prefer_1st_tmin)
#%%
sns.histplot(pr_prefer_1st)
plt.title('p_boltz')

#%%
def build_preference_model():
    glorot = keras.initializers.GlorotUniform(seed=seed)
    x1_input = keras.Input(shape=(28,28,1))
    x2_input = keras.Input(shape=(28,28,1))
    temp_input = keras.Input(shape=(1,))

    conv_base = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer=glorot),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer=glorot),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer=glorot),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer=glorot),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5, seed=seed),
        # keras.layers.Dense(32, activation="relu", dtype="float32", kernel_initializer=glorot),
        # keras.layers.Dropout(0.5, seed=seed),
        keras.layers.Dense(1, activation="sigmoid", dtype="float32", kernel_initializer=glorot),
    ])

    u1 = conv_base(x1_input)
    u2 = conv_base(x2_input)

    max_utility = keras.ops.maximum(u1, u2)
    exp_diff1 = keras.ops.exp((u1 - max_utility) / temp_input)
    exp_diff2 = keras.ops.exp((u2 - max_utility) / temp_input)
    pr_prefer_u1 = exp_diff1 / (exp_diff1 + exp_diff2)

    model = keras.Model(
        inputs=[x1_input, x2_input, temp_input],
        outputs=pr_prefer_u1,
    )
    utility_model = keras.Model(
        inputs=model.input,
        outputs=[u1, u2],
    )
    return model, utility_model

#%% Build a model using the ground-truth temperature data
true_temp_model = build_preference_model()
true_temp_model[0].compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
    ],
)

true_temp_model[0].fit(
    x=[x1_train, x2_train, temps],
    y=prefs,
    batch_size=32,
    epochs=1,
)

#%% Let's try to build a "perfectly rational" preference model, using a fixed (low) temperature
fixed_temps = np.ones_like(temps) * temps.min()
rational_model = build_preference_model()
rational_model[0].compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
    ],
)
rational_model[0].fit(
    x=[x1_train, x2_train, fixed_temps],
    y=prefs,
    batch_size=32,
    epochs=1,
)

#%% Skyline: if prefs actually used T=Tmin
pr_prefer_1st_skyline = boltzmann_probability(u1_train, u2_train, fixed_temps)
prefs_skyline = np.random.binomial(1, pr_prefer_1st_skyline)

skyline_model = build_preference_model()
skyline_model[0].compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
    ],
)
skyline_model[0].fit(
    x=[x1_train, x2_train, fixed_temps],
    y=prefs_skyline,
    batch_size=32,
    epochs=1,
)

#%% Build a model using only the low-temperature data
low_temp_model = build_preference_model()
low_temp_model[0].compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
    ],
)

percentile = 0.1
temp_cutoff = sorted(temps)[int(N*percentile)]
low_temp_idx = temps < temp_cutoff

low_temp_model[0].fit(
    x=[x1_train[low_temp_idx], x2_train[low_temp_idx], temps[low_temp_idx]],
    y=prefs[low_temp_idx],
    batch_size=32,
    epochs=1,
)

#%%

gt_utilities_model = build_preference_model()
gt_utilities_model[1].compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.MeanSquaredError(name="mse"),
        keras.metrics.MeanSquaredError(name="mse"),
    ],
)
gt_utilities_model[1].fit(
    x=[x1_train, x2_train, fixed_temps],
    y=[u1_train, u2_train],
    batch_size=32,
    epochs=1,
)

#%% Evaluate a model
def mean_absolute_error(y_true, y_pred):
    return keras.ops.mean(keras.ops.abs(y_true - y_pred))

def kl_divergence(y_true, y_pred):
    vmin = np.zeros_like(y_pred)
    vmin[y_pred==0.5] = -np.log(0.5)
    return keras.ops.binary_crossentropy(target=y_true, output=y_pred) - vmin

def plot_model_probs(model, model_name, ax=None, type='hist'):
    pref_classes = {
        '>': u1_test > u2_test,
        '~': u1_test == u2_test,
        '<': u1_test < u2_test,
    }
    fake_temps = Tmin * np.ones(len(x1_test))
    pr_prefer_left = model[0].predict((x1_test, x2_test, fake_temps), batch_size=32)

    vmin = pr_prefer_left.min()
    vmax = pr_prefer_left.max()
    bins = np.linspace(vmin, vmax, 30)

    pref_predictions = {
        key: pr_prefer_left[idx].squeeze() for key, idx in pref_classes.items()
    }

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Probability')
    if type == 'hist':
        sns.histplot(pref_predictions, alpha=0.7, palette=['C0', 'C8', 'C3'], bins=bins, kde=False, stat="count", ax=ax, legend=True)
        ax.set_ylabel('Count')
        ax.get_legend().set_loc('upper center' if 'rational' not in model_name else 'best')
    elif type == 'kde':
        sns.kdeplot(pref_predictions, alpha=0.7, palette=['C0', 'C8', 'C3'], ax=ax, common_norm=False, clip=[0,1])
        ax.set_ylabel('Density')
    else:
        raise ValueError(f"Unknown 'type' parameter: {type}")
    if should_show_plot:
        ax.show()

models = {
    # 'skyline_model': skyline_model,
    'rational_model': rational_model,
    'true_temp_model': true_temp_model,
    # 'low_temp_model': low_temp_model,
    'gt_utilities_model': gt_utilities_model,
}
for type in ['hist', 'kde']:
    fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True)
    for (model_name, model), ax in zip(models.items(), axes.flatten()):
        plot_model_probs(model, model_name, ax=ax, type=type)
    plt.tight_layout()
    plt.show()


#%%

models = {
    # 'skyline_model': skyline_model,
    'rational_model': rational_model,
    'true_temp_model': true_temp_model,
    # 'low_temp_model': low_temp_model,
    'gt_utilities_model': gt_utilities_model,
}
def plot_utility_calibration(model, model_name, ax=None):
    fake_temps = Tmin * np.ones(len(x1_test))
    u1_hat, _ = model[1].predict((x1_test, x2_test, fake_temps), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Actual utility')
    sns.violinplot(x=9*u1_test, y=9*u1_hat.squeeze(), ax=ax, legend=True)
    # ax.set_ylim([0,9])
    ax.set_ylabel('Predicted utility')
    if should_show_plot:
        ax.show()


fig, axes = plt.subplots(1,3, figsize=(10, 3), sharex=True)
for (model_name, model), ax in zip(models.items(), axes.flatten()):
    plot_utility_calibration(model, model_name, ax=ax)
plt.tight_layout()
plt.show()

#%%

models = {
    # 'skyline_model': skyline_model,
    'rational_model': rational_model,
    'true_temp_model': true_temp_model,
    # 'low_temp_model': low_temp_model,
    'gt_utilities_model': gt_utilities_model,
}
def plot_probability_calibration(model, model_name, ax=None):
    test_temps = 10**np.random.uniform(low=-2., high=2., size=len(x1_test)).astype(np.float32)
    test_pr_prefer_1st = boltzmann_probability(u1_test, u2_test, test_temps)
    pr_boltz_hat = model[0].predict((x1_test, x2_test, test_temps), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel(r'Actual $p_{boltz}$')
    sns.kdeplot(x=test_pr_prefer_1st, y=pr_boltz_hat.squeeze(), clip=[0,1], ax=ax, legend=True)
    ax.set_ylim([0,1])
    ax.set_ylabel(r'Predicted $\hat p_{boltz}$')
    if should_show_plot:
        ax.show()


fig, axes = plt.subplots(1,3, figsize=(10, 4), sharex=True)
for (model_name, model), ax in zip(models.items(), axes.flatten()):
    plot_probability_calibration(model, model_name, ax=ax)
plt.tight_layout()
plt.show()
