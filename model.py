import backend
import keras

def build_preference_model(seed):
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

def load_model(models_dir, name):
    # Load a trained prefences/utilities model pair for inference
    prefs_model = keras.saving.load_model(models_dir+f'prefs/{name}.keras')
    utils_model = keras.saving.load_model(models_dir+f'utils/{name}.keras')
    return (prefs_model, utils_model)
