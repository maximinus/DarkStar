#!/usr/bin/env python3

import sys
from keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
import keras.backend as K
from keras import regularizers
from keras.layers import Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from tensorflow import set_random_seed
import numpy as np

from darkstar.helpers import *
from darkstar.recorder import Recorder
from tqdm import tqdm
import random


# 10 seconds of 22.05kHz
AUDIO_LENGTH = 22050 * 10
NUMBER_OF_CLASSES = 2
RANDOM_SEED = 1234
BATCH_SIZE = 16

# how many of the samples to use for testing
TEST_SET = 0.2

# how may files in total to use
# set to -1 for all files
MAX_FILES = 20


def getModel():
    model = Sequential()
    model.add(Conv1D(128, input_shape=[AUDIO_LENGTH, 1], kernel_size=8, strides=4,
                     padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    model.add(Conv1D(128, kernel_size=3, strides=1,
                     padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    model.add(Conv1D(256, kernel_size=3, strides=1,
                     padding='same', kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    model.add(Conv1D(512,kernel_size=3, strides=1, padding='same',
                     kernel_initializer='glorot_uniform',
                     kernel_regularizer=regularizers.l2(l=0.0001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    # K is Keras.backend
    model.add(Lambda(lambda x: K.mean(x, axis=1)))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    return(model)


def initRandomSeeds():
    # start with same seed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    set_random_seed(RANDOM_SEED)


class DataSet:
    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.x_train = xtrain
        self.y_train = ytrain
        self.x_test = xtest
        self.y_test = ytest


def getData():
    # we need an array of questions, and one of answers
    sbd_source = getDataDirectory('PROCESSED_WAV/GD_NUMPY')
    aud_source = getDataDirectory('PROCESSED_WAV/OTHER_NUMPY')
    questions = []

    print('Loading Grateful Dead files')
    for file in tqdm(getAllFiles(sbd_source, extension='npy')[:MAX_FILES]):
        sound_data = np.load(file)
        questions.append([sound_data, 1.0])

    print('Loading non-Dead files')
    for file in tqdm(getAllFiles(aud_source, extension='npy')[:MAX_FILES]):
        sound_data = np.load(file)
        questions.append([sound_data, 0.0])

    print('\nLoaded {0} tests\n'.format(len(questions)))

    # shuffle and split
    random.shuffle(questions)
    data_split = int(len(questions) * TEST_SET)
    test_data = questions[:data_split]
    train_data = questions[data_split:]

    x_train = np.array([x[0] for x in train_data])
    y_train = np.array([x[1] for x in train_data])
    x_test = np.array([x[0] for x in test_data])
    y_test = np.array([x[1] for x in test_data])
    
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    y_train = to_categorical(y_train, NUMBER_OF_CLASSES)
    y_test = to_categorical(y_test, NUMBER_OF_CLASSES)

    return(DataSet(x_train, y_train, x_test, y_test))


def generateData():
    while True:
        # constantly yields a collection of data
        pass


if __name__ == '__main__':
    initRandomSeeds()
    data = getData()

    # then we need the model
    model = getModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    model.fit(data.x_train,
              data.y_train,
              batch_size=BATCH_SIZE,
              epochs=100,
              verbose=1,
              shuffle=True,
              validation_data=(data.x_test, data.y_test),
              callbacks=[Recorder()])
