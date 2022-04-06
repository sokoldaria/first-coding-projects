# Project CNN model

import numpy as np
import pandas as pd
import seaborn as sns

import gzip
import numpy as np

# Code aufbauend auf: https://stackoverflow.com/a/62781370
def load_images(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count, column_count))
        return images
    
def load_labels(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels

X_train = load_images("train-images-idx3-ubyte.gz")
y_train = load_labels("train-labels-idx1-ubyte.gz")

X_test = load_images("t10k-images-idx3-ubyte.gz")
y_test = load_labels("t10k-labels-idx1-ubyte.gz")

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

# Tensorflow laden
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    keras.Input(shape = (28, 28, 1)),
    layers.Conv2D(10, kernel_size = (3, 3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(10, kernel_size = (3, 3), padding = "same", activation = "relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(100, activation = "relu"),
    layers.Dense(10, activation = "softmax")
])

print(model.summary())

model.compile(
    optimizer = keras.optimizers.RMSprop(),
    loss = keras.losses.CategoricalCrossentropy(),
    metrics = [
        keras.metrics.CategoricalAccuracy()
    ]
)

model.fit(X_train, y_train, 
          batch_size = 64, 
          epochs = 10, 
          validation_split = 0.1)

model.fit(X_train, y_train, 
          batch_size = 64, 
          epochs = 10, 
          validation_split = 0.1)

y_test_pred = model.predict(X_test)

print(np.mean(np.argmax(y_test_pred, axis = 1) == y_test))