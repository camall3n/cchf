import os
os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% Set seeds
seed = 42
np.random.seed(seed)

#%% Import MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#%% Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

#%% Scale labels to the [0, 1] range
y_train = y_train.astype("float32") / 9
y_test = y_test.astype("float32") / 9

#%% Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

#%% Generate (input, input) -> (output, output) examples

# The (x1, x2) values consitute a comparison between two inputs, and the y values
# are the corresponding ground truth utilities for the individual inputs.
# Preferences are assumed to be Boltzmann distributed relative to these utilities,
# with varying temperatures.

# The goal is to train a model to predict the utilities from the inputs and preferences

# Generate random indexes for the first and second inputs
N = 60000
# idx1 = np.random.randint(0, x_train.shape[0], size=(N,))
# idx2 = np.random.randint(0, x_train.shape[0], size=(N,))
idx1 = np.arange(N)
idx2 = np.arange(N)
np.random.shuffle(idx1)
np.random.shuffle(idx2)

# Select input pairs and corresponding labels
x_train1 = x_train[idx1]
x_train2 = x_train[idx2]
utility_train1 = y_train[idx1]
utility_train2 = y_train[idx2]

def boltzmann_probability(utility1, utility2, temp):
    max_utility = np.maximum(utility1, utility2)
    # Subtract the max utility to prevent overflow
    exp_diff1 = np.exp((utility1 - max_utility) / temp)
    exp_diff2 = np.exp((utility2 - max_utility) / temp)
    return exp_diff1 / (exp_diff1 + exp_diff2)

#%%
temps_easy = 10**np.random.uniform(low=-2., high=-2., size=N).astype(np.float32)
temps_hard = 10**np.random.uniform(low=-2., high=2., size=N).astype(np.float32)
temps = temps_easy
pr_prefer_1st = boltzmann_probability(utility_train1, utility_train2, temps)

# Sample preferences from bernoulli distribution
prefs = np.random.binomial(1, pr_prefer_1st)
sns.histplot(pr_prefer_1st)
plt.title('Pr(U(x1) > U(x2))')

#%%
def build_preference_model():
    x1_input = keras.Input(shape=(28,28,1))
    x2_input = keras.Input(shape=(28,28,1))
    temp_input = keras.Input(shape=(1,))

    conv_base = keras.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])

    u1 = conv_base(x1_input)
    u2 = conv_base(x2_input)

    max_utility = keras.ops.maximum(u1, u2)
    exp_diff1 = keras.ops.exp((u1 - max_utility) / temp_input)
    exp_diff2 = keras.ops.exp((u2 - max_utility) / temp_input)
    pr_prefer_u1 = exp_diff1 / (exp_diff1 + exp_diff2)

    model = keras.Model(
        inputs=[x1_input, x2_input, temp_input],
        outputs=pr_prefer_u1
    )
    return model

model = build_preference_model()
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc"),
    ],
)

model.fit(
    x=[x_train1, x_train2, temps],
    y=prefs,
    batch_size=32,
    epochs=1,
)

#%%
for i in range(100):
    plt.imshow(np.concatenate([x_train1[i], x_train2[i]], axis=1))
    # print([utility_train1[0], utility_train2[0]], temps[0])
    pr_prefer_left = model([x_train1[i:i+1], x_train2[i:i+1], temps[i:i+1]])
    plt.title(f'Pr(U(left) > U(right)) = {pr_prefer_left[0,0]}')
    plt.show()
