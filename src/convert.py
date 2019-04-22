#!/usr/bin/env python3

import os
import sys
import numpy
import shutil
import random
from pydub import AudioSegment

from helpers import getDataDirectory, sliceAndConvert
from constants import TIMESLICE, WAV_FOLDER, MEL_FOLDER


def convertDataToMel():
	print('* Converting GD to MEL spectrograms...')
	wav_directory = getDataDirectory('WAV/GRATEFUL_DEAD')
	mel_directory = getDataDirectory('MEL/GRATEFUL_DEAD')
	sliceAndConvert(wav_directory, mel_directory)

	print('* Converting Other to MEL spectrograms...')
	wav_directory = getDataDirectory('WAV/OTHER')
	mel_directory = getDataDirectory('MEL/OTHER')
	sliceAndConvert(wav_directory, mel_directory)


if __name__ == '__main__':
	convertDataToMel()
