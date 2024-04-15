from argparse import Namespace

import backend
import keras
import numpy as np

from util import boltzmann_probability

def generate_pref_pairs_dataset(x, y, N=None, tmin=1e-2, tmax=1e2):
    """Generate (input, input) -> (output, output) examples from MNIST (x, y) data.
    If N is None, generate one pair for each entry in the original MNIST dataset.
    Otherwise, generate N pairs total."""

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
        assert N >= 1, 'Dataset must contain at least 1 sample'
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
    true_prefs = (np.sign(u1 - u2) + 1) / 2
    rng = np.random.default_rng()
    temps = np.e**rng.uniform(low=np.log(tmin), high=np.log(tmax), size=len(x1))#.astype(np.float32)
    # 1 corresponds to preferring A
    p_boltz_2ary = boltzmann_probability((u1, u2), temps)[:,0]
    y_boltz_2ary = rng.binomial(1, p_boltz_2ary)

    u_indifferent = (u1 + u2) / 2
    # flip choice order so that 1 still corresponds to preferring A
    p_boltz_3ary = boltzmann_probability((u2, u_indifferent, u1), temps)
    y_boltz_3ary = np.argmax(rng.multinomial(1, p_boltz_3ary), axis=-1) / 2

    return Namespace(**{
        'x1': x1,
        'x2': x2,
        'u1': u1,
        'u2': u2,
        'T': temps,
        'p_boltz_2ary': p_boltz_2ary,
        'y_boltz_2ary': y_boltz_2ary,
        'p_boltz_3ary': p_boltz_3ary,
        'y_boltz_3ary': y_boltz_3ary,
        'y': true_prefs,
    })

def generate_dataset(n_train=None, n_test=None, tmin=1e-2, tmax=1e2):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    train = generate_pref_pairs_dataset(x_train, y_train, n_train, tmin, tmax)
    test = generate_pref_pairs_dataset(x_test, y_test, n_test, tmin, tmax)
    return train, test

def filter_dataset_by_temp(dataset, tmax):
    low_temp_idx = dataset.T <= tmax
    data_dict = dataset.__dict__.copy()
    for key, val in data_dict.items():
        data_dict[key] = val[low_temp_idx]
    return Namespace(**data_dict)
