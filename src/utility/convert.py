#!/usr/bin/env python3

import os
import sys
import numpy
import shutil
import random
from pydub import AudioSegment

from helpers import getDataDirectory, sliceAndConvert
from constants import TIMESLICE, WAV_FOLDER, MEL_FOLDER


def loadAsArray(files, value):
	data = []
	dtype = numpy.dtype('b')
	for i in files:
		raw_array = numpy.fromfile(i, dtype)
		converted = raw_array.astype(float)
		# range from 0 to 1
		converted += 128
		converted /= 255
		# make sure the data is correct
		assert min(converted) >= 0.0
		assert max(converted) <= 1.0
		data.append([converted, value])
	return data


def saveData(all_data):
	random.shuffle(all_data)
	x = numpy.array([i[0] for i in all_data])
	y = numpy.array([i[1] for i in all_data])
	print('  Learning shape:', x.shape)
	print('  Question shape:', y.shape)
	# save these arrays
	numpy.save('X_DATA', x)
	numpy.save('Y_DATA', y)	


def convertInput():
	print('Building data')
	files = getAllFiles('./SLICES/{0}'.format(GRATEFUL_DEAD), 'raw')
	data_yes = loadAsArray(files, 1.0)
	files = getAllFiles('./SLICES/{0}'.format(OTHER), 'raw')
	data_no = loadAsArray(files, 0.0)

	# make sure there is a 50/50 mix of YES and NO
	print('  GD Data: {0}'.format(len(data_yes)))
	print('  Other Data: {0}'.format(len(data_no)))
	shortest = min(len(data_yes), len(data_no))
	print('  Shortest: {0}'.format(shortest))
	all_data = data_yes[:shortest] + data_no[:shortest]
	print('  Size of all data: {0}'.format(len(all_data)))
	saveData(all_data)


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
