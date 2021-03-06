#!/usr/bin/env python

# The audio files are stored this way:
# 4096 samples per second, 2 seconds of audio, 2 bytes per sample
# 4096 * 2 * 2 = 16384
# our input shape is related to this:
# 4096 * 2 -> 8192
# In this experiment, the input is a 4096 array with 16-bit values -> 8192 bytes
# this gets turned into floats, so an array of 4096 floats

# try and reduce console noise
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from tqdm import tqdm
import random
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, Activation, MaxPooling1D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

from darkstar import helpers

TOTAL_CLASSES = 1
TOTAL_EPOCHS = 20
BATCH_SIZE = 1

AUD_SLICES = 'music/slices/aud'
SBD_SLICES = 'music/slices/sbd'

LOG_DIRECTORY = 'logs/'


class ModelData:
    CHECK_RATIO = 0.2

    def __init__(self, aud, sbd):
        # we need to the class data to the arrays
        all_data = [[x, 0] for x in aud]
        all_data.extend([[x, 1] for x in sbd])
        # shuffle in place
        random.shuffle(all_data)
        # lop off so it's a multiple of 4
        all_data = all_data[:-(len(all_data) % 4)]
        # then split data into groups
        check = int(len(all_data) * self.CHECK_RATIO)
        array = np.array([x[0] for x in all_data[check:]])
        self.x_test = np.expand_dims(array, 2)
        #self.x_test = array
        self.y_test = np.array([x[1] for x in all_data[check:]])
        self.y_test = self.y_test.astype(np.uint8)

        array = np.array([x[0] for x in all_data[:check]])
        self.x_check = np.expand_dims(array, 2)
        #self.x_check = array
        self.y_check = np.array([x[1] for x in all_data[:check]])
        self.y_check = self.y_check.astype(np.uint8)

    def __repr__(self):
        atext = f'[{len(self.check_aud)}, {len(self.test_aud)}]'
        stext = f'[{len(self.check_sbd)}, {len(self.test_sbd)}]'
        return f'ModelData: [{len(self.x_test)}, {len(self.x_check)}]'


def getLoggerCallback():
    logdir = LOG_DIRECTORY + datetime.now().strftime("%Y%m%d-%H%M%S")
    return TensorBoard(log_dir=logdir)

def loadAllFiles(folder):
    # ensure folder exists
    if not os.path.isdir(folder):
        helpers.logError(f'{folder} is not a valid folder')
    files = helpers.getAllFiles(folder, extension='raw')
    audio = []  
    print('Loading data')
    for i in tqdm(files):
        audio_data = np.fromfile(i, dtype=np.uint16, count=-1, sep='', offset=0)
        # convert to a float
        float_data = audio_data.astype(np.float32)
        # then convert to the range 0 -> 1
        float_data /= 65536
        audio.append(float_data)
    return audio

def getData():
    # load everything as a dataset, since it will fit into memory
    root_folder = helpers.getRootDirectory()
    aud_files = loadAllFiles(os.path.join(root_folder, AUD_SLICES))
    sbd_files = loadAllFiles(os.path.join(root_folder, SBD_SLICES))
    return ModelData(aud_files, sbd_files)

def getDiscriminator():
    model = tf.keras.models.Sequential()
    model.add(Conv1D(32, input_shape=(8192, 1), kernel_size=2, dilation_rate=1, strides=1, padding="SAME"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=2, dilation_rate=2, strides=1, padding="SAME"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=2, dilation_rate=4, strides=1, padding="SAME"))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    data = getData()
    model = getDiscriminator()
    model.summary()
    results = model.fit(data.x_test, data.y_test,
                        validation_data=(data.x_check, data.y_check),
                        epochs=TOTAL_EPOCHS,
                        callbacks=[getLoggerCallback()])
