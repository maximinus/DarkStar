#!/usr/bin/env python3

import os
import sys
import numpy
import shutil
import random
from pydub import AudioSegment

from helpers import getDataDirectory, sliceFiles
from constants import TIMESLICE, DATA_SIZE


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


def sliceData():
	# we go through the WAV directory and fill up the SLICES directory
	# we use a similar directory structure
	wav_directory = getDataDirectory('WAV')
	slice_directory = getDataDirectory('SLICE')
	print('* Slicing WAV files...')
	sliceFiles(wav_directory, slice_directory)


def createMelData():
	# we got through folders and convert all the sound clips into MEL spectrograms
	# we mimic the directory structure for all the images
	# we delete the output directory at the start of the process
	# the root directory is always the SLICES directory
	pass


if __name__ == '__main__':
	sliceData()
