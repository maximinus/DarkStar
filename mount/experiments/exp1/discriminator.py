#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout
import numpy as np

import darkstar.helpers

TOTAL_CLASSES = 1

def getData():
    # return the data our network requires
    # for 4 items we need a shape of (4, 1) -> 4 items of length 1 each
    # NOTE: to do this with an array you need np.array([[1], [1], [1], [1]])
    data = np.ones((12, 1))
    return(data)

def getDiscriminator():
    model = tf.keras.models.Sequential()
    # let's just have 1 single filter with a large skip
    model.add(Conv1D(64, input_shape=(4096, 1), kernel_size=2, dilation_rate=1, padding="SAME"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(TOTAL_CLASSES, activation='relu'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data = getData()
    print(f'Data shape: {data.shape}')
    model = getDiscriminator()
    model.summary()
