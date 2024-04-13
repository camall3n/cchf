import backend
import keras

def mean_absolute_error(y_true, y_pred):
    return keras.ops.mean(keras.ops.abs(y_true - y_pred))

def kl_divergence(y_true, y_pred):
    vmin = keras.zeros_like(y_pred)
    vmin[y_pred==0.5] = -keras.log(0.5)
    return keras.ops.binary_crossentropy(target=y_true, output=y_pred) - vmin
