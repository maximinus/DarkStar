#!/usr/bin/env python

# The audio files are stored this way:
# 4096 samples per second, 2 seconds of audio, 2 bytes per sample
# 4096 * 2 * 2 = 16384
# In this experiment, the input is a 4096 array with 16-bit values -> 8192 bytes
# this gets turned into floats, so an array of 4096 floats

# try and reduce console noise
import warnings
warnings.filterwarnings("ignore")

import os
import sys
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Dropout
import numpy as np

from darkstar import helpers

TOTAL_CLASSES = 1

AUD_SLICES = 'music/slices/aud'
SBD_SLICES = 'music/slices/sbd'


class ModelData:
    CHECK_RATIO = 0.2

    def __init__(self, aud, sbd):
        # randomise the 2 arrays and split up
        random.shuffle(aud)
        random.shuffle(sbd)
        check = int(len(aud) * self.CHECK_RATIO)
        self.test_aud = aud[check:]
        self.check_aud = aud[:check]
        check = int(len(sbd) * self.CHECK_RATIO)
        self.test_sbd = sbd[check:]
        self.check_sbd = sbd[:check]

    def __repr__(self):
        atext = f'[{len(self.check_aud)}, {len(self.test_aud)}]'
        stext = f'[{len(self.check_sbd)}, {len(self.test_sbd)}]'
        return f'ModelData: Aud {atext}, Sbd {stext}'


def loadAllFiles(folder):
    # ensure fodler exists
    if not os.path.isdir(folder):
        helpers.logError(f'{folder} is not a valid folder')
    files = helpers.getAllFiles(folder, extension='raw')
    audio = []
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
    print(data)
    sys.exit()

    print(f'Data shape: {data.shape}')
    model = getDiscriminator()
    model.summary()
