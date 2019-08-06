#!/usr/bin/env python3

import sys
from tqdm import tqdm
import numpy as np
from darkstar.helpers import *

# How we want the data
# the data has to be a WAV file

# the format we require is a MONO 22.05kHz 16-bit 8 second file
# this is 22050 * 2 * 10 bytes = 441000

def sliceFilesAndConvert():
	# get the data directory
	gd_files = getDataDirectory('WAV/GRATEFUL_DEAD')
	ng_files = getDataDirectory('WAV/OTHER')
	slice_dir = getDataDirectory('SLICES')
	gd_out = getDataDirectory('PROCESSED_WAV/GD')
	ng_out = getDataDirectory('PROCESSED_WAV/OTHER')

	# clear all output directories
	clearDirectory(gd_out)
	clearDirectory(ng_out)
	clearDirectory(slice_dir)

	print('Slicing GD')
	index = 0
	for gd_file in tqdm(getAllFiles(gd_files)):
		index = sliceAsWav(gd_file, slice_dir, index)

	print('Converting GD')
	index = 0
	for gd_file in tqdm(getAllFiles(slice_dir)):
		# 22.05kHz mono 16-bit
		convertWavFormat(gd_file, gd_out, 22050, 1, 2, index)
		index += 1
	clearDirectory(slice_dir)

	print('Slicing Other')
	index = 0
	for ng_file in tqdm(getAllFiles(ng_files)):
		index = sliceAsWav(ng_file, slice_dir, index)
	print('Converting Other')
	index = 0
	for ng_file in tqdm(getAllFiles(slice_dir)):
		# 22.05kHz mono 16-bit
		convertWavFormat(ng_file, ng_out, 22050, 1, 2, index)
		index += 1
	# cleanup unused files
	clearDirectory(slice_dir)


def convertToNumpy():
	gd_in = getDataDirectory('PROCESSED_WAV/GD')
	ng_in = getDataDirectory('PROCESSED_WAV/OTHER')	
	gd_out = getDataDirectory('PROCESSED_WAV/GD_NUMPY')
	ng_out = getDataDirectory('PROCESSED_WAV/OTHER_NUMPY')

	# clear output directories
	clearDirectory(gd_out)
	clearDirectory(ng_out)
	
	print('Converting SBD to Numpy from {0}'.format(gd_in))
	index = 0
	for filename in tqdm(getAllFiles(gd_in, extension='raw')):
		# load
		np_data = np.fromfile(filename, dtype='int16')
		# convert to float
		float_data = np_data.astype(float)
		# normalise from -32768 -> +32767 => 0 -> 1
		float_data += 32768
		float_data /= 65535
		assert ((float_data <= 1.0).all() and (float_data >= 0.0).all())
		# now we save the data
		save_file = '{0}/{1}.npy'.format(gd_out, getStringName(index))
		np.save(save_file, float_data)
		index += 1
	print('Written {0} files to {1}'.format(index, gd_out))

	print('Converting AUD to Numpy from {0}'.format(ng_in))
	index = 0
	for filename in tqdm(getAllFiles(ng_in, extension='raw')):
		# load
		np_data = np.fromfile(filename, dtype='int16')
		# convert to float
		float_data = np_data.astype(float)
		# normalise
		float_data += 32768
		float_data /= 65535
		assert ((float_data <= 1.0).all() and (float_data >= 0.0).all())
		# now we save the data
		save_file = '{0}/{1}.npy'.format(ng_out, getStringName(index))
		np.save(save_file, float_data)
		index += 1
	print('Written {0} files to {1}'.format(index, ng_out))


if __name__ == '__main__':
	sliceFilesAndConvert()
	convertToNumpy()
