#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from keras import regularizers
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.core import Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from darkstar.helpers import *

# 1: Extract the data for a sound file in the format we use and draw it to a graph
# 2: Add a convolutional network and draw the output of this network
# 3: Show how the weights work and how many / their values

# 10 seconds of 22.05kHz -> 220500 samples
SOUNDFILE = '000015.npy'

# we only want a limited number of these
GRAPH_LENGTH = 500
AUDIO_LENGTH = 22050 * 10


def getLayer():
    return Conv1D(128,
                  input_shape=[AUDIO_LENGTH, 1],
                  kernel_size=10,
                  strides=4,
                  padding='same',
                  kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(l=0.0001))


def getModel():
    # returns the model we use
    model = Sequential()
    model.add(getLayer())
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=None))
    return model


def loadFile():
    sbd_source = getDataDirectory('PROCESSED_WAV/SBD_NUMPY')
    file_source = '{0}/{1}'.format(sbd_source, SOUNDFILE)
    return np.load(file_source)


def drawGraph(sound1, sound2):
    plt.plot(sound1, color='red')
    plt.plot(sound2, color='green')
    plt.show()


def plotModelDifference():
    model = getModel()
    model.summary()

    sound = loadFile()
    # add dimensions so that we are [AUDIO, 1]
    sound_batch = np.expand_dims(sound, axis=1)
    # create an array of these, i.e. add another dimension
    sound_batch = np.expand_dims(sound_batch, axis=0)    
    # run through the model
    conv_sound = model.predict(sound_batch)
    # reduce to one sample
    sound_reduce = np.squeeze(conv_sound, axis=0)
    # now we have (55125, 128) shape -> the output of the model. Swap the axis
    sound_reduce = np.swapaxes(sound_reduce, 0, 1)
    # and take the first
    final_sound = sound_reduce[0]
    drawGraph(final_sound[:GRAPH_LENGTH], sound[:GRAPH_LENGTH])


if __name__ == '__main__':
    plotModelDifference()
