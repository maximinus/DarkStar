#!/usr/bin/env python3

import sys
from tqdm import tqdm
import numpy as np
from darkstar.helpers import *

# How we want the data
# the data has to be a WAV file

# the format we require is a MONO 22.05kHz 16-bit 8 second file
# this is 22050 * 2 * 10 bytes = 441000

MAX_FILES = 2000

def sliceFileAndConvert():
	# get the data directory
	sbd = getDataDirectory('ORIGINAL/SBD_AUD/SBD')
	aud = getDataDirectory('ORIGINAL/SBD_AUD/AUD')
	slice_dir = getDataDirectory('SLICES')
	sbd_out = getDataDirectory('PROCESSED_WAV/SBD')
	aud_out = getDataDirectory('PROCESSED_WAV/AUD')

	clearDirectory(sbd_out)
	clearDirectory(aud_out)
	clearDirectory(slice_dir)

	print('Slicing SBDS')
	index = 0
	for sbd_file in tqdm(getAllFiles(sbd)):
		index = sliceAsWav(sbd_file, slice_dir, index)

	print('Converting SBDS')
	index = 0
	for sbd_file in tqdm(getAllFiles(slice_dir)):
		# 22.05kHz mono 16-bit
		convertWavFormat(sbd_file, sbd_out, 22050, 1, 2, index)
		index += 1
	clearDirectory(slice_dir)

	print('Slicing AUDS')
	index = 0
	for aud_file in tqdm(getAllFiles(aud)):
		index = sliceAsWav(aud_file, slice_dir, index)
	print('Converting AUDS')
	index = 0
	for aud_file in tqdm(getAllFiles(slice_dir)):
		# 22.05kHz mono 16-bit
		convertWavFormat(aud_file, aud_out, 22050, 1, 2, index)
		index += 1
	clearDirectory(slice_dir)


def convertToNumpy():
	sbd_in = getDataDirectory('PROCESSED_WAV/SBD')
	aud_in = getDataDirectory('PROCESSED_WAV/AUD')	
	sbd_out = getDataDirectory('PROCESSED_WAV/SBD_NUMPY')
	aud_out = getDataDirectory('PROCESSED_WAV/AUD_NUMPY')
	clearDirectory(sbd_out)
	clearDirectory(aud_out)
	
	print('Converting SBD to Numpy from {0}'.format(sbd_in))
	index = 0
	for filename in tqdm(getAllFiles(sbd_in, extension='raw')[:MAX_FILES]):
		# load
		np_data = np.fromfile(filename, dtype='int16')
		# convert to float
		float_data = np_data.astype(float)
		# normalise from -32768 -> +32767 => 0 -> 1
		float_data += 32768
		float_data /= 65535
		assert ((float_data <= 1.0).all() and (float_data >= 0.0).all())
		# now we save the data
		save_file = '{0}/{1}.npy'.format(sbd_out, getStringName(index))
		np.save(save_file, float_data)
		index += 1
	print('Written {0} files to {1}'.format(index, sbd_out))

	print('Converting AUD to Numpy from {0}'.format(aud_in))
	index = 0
	for filename in tqdm(getAllFiles(aud_in, extension='raw')[:MAX_FILES]):
		# load
		np_data = np.fromfile(filename, dtype='int16')
		# convert to float
		float_data = np_data.astype(float)
		# normalise
		float_data += 32768
		float_data /= 65535
		assert ((float_data <= 1.0).all() and (float_data >= 0.0).all())
		# now we save the data
		save_file = '{0}/{1}.npy'.format(aud_out, getStringName(index))
		np.save(save_file, float_data)
		index += 1
	print('Written {0} files to {1}'.format(index, aud_out))


if __name__ == '__main__':
	#sliceAndConvert()
	convertToNumpy()
