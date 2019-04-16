#!/usr/bin/env python3

import os
import sys
import numpy
import shutil
import random
from pydub import AudioSegment
from helpers import getAllFiles, Slicer, TIMESLICE
from constants import TIMESLICE, DATA_SIZE


GRATEFUL_DEAD = 'GRATEFUL_DEAD'
OTHER = 'OTHER'
WAVS = 'WAV'
SLICES = 'SLICES'


def slice():
	print('  Slicing GD files...')
	wav_files = getAllFiles('./WAV/{0}'.format(GRATEFUL_DEAD))
	print('    (Found {0} GD files)'.format(len(wav_files)))
	foo = Slicer(wav_files, GRATEFUL_DEAD)
	gd_total = foo.sliceFiles()

	print('  Slicing Other files...')
	wav_files = getAllFiles('./WAV/{0}'.format(OTHER))
	print('    (Found {0} other files)'.format(len(wav_files)))
	foo = Slicer(wav_files, OTHER)
	other_total = foo.sliceFiles()
	print('  GD: {0}, Other: {1}\n  Total Files: {2}'.format(gd_total, other_total, gd_total + other_total))


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


def getEmpty(size, answer):
	# create an array of the right size
	final_data = []
	for i in range(size):
		new_data = [random.uniform(0.0, 1.0) for x in range(DATA_SIZE)]
		final_data.append([numpy.array(new_data), answer])
	return final_data


def buildEmpty():
	# build a set of 'NO' data that is always 0.1 or less
	# we assume the 'YES' data is already done
	print('Building data with empty NO')
	files = getAllFiles('./SLICES/{0}'.format(GRATEFUL_DEAD), 'raw')
	data_yes = loadAsArray(files, 1.0)
	data_no = getEmpty(len(data_yes), 0.0)
	all_data = data_yes + data_no
	saveData(all_data)


def buildRandom():
	# build a set of 'NO' data that is random
	# we assume the 'YES' data is already done
	pass


if __name__ == '__main__':
	if len(sys.argv) < 2:
		slice()
		convertInput()
		sys.exit(True)
	if sys.argv[1] == 'slice':
		slice()
		sys.exit(True)
	if sys.argv[1] == 'build':
		convertInput()
		sys.exit(True)
	if sys.argv[1] == 'empty':
		buildEmpty()
		sys.exit(True)
	print('  Error: No such command {0}'.format(sys.argv[1]))
