import os

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=(100,)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(256),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(2)
])
model.save("fc_model")
