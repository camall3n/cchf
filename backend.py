import os
import warnings

os.environ["KERAS_BACKEND"] = "jax"
warnings.filterwarnings("ignore", category=UserWarning, message='Some donated buffers were not usable')
