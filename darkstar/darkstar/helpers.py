#!/usr/bin/env python3

import os
import shutil
from pydub import AudioSegment

# length of each slice in milliseconds
TIMESLICE = 10000
# difference of time between samples
TIME_DELTA = 4000

SLICES_FOLDER = 'SLICES'
MEL_FOLDER = 'MEL'
TMP_FOLDER = 'TMP'
WAV_FOLDER = 'WAV'


def logError(message):
	print(f'  Error: {message}')
	sys.exit(False)


def getRootDirectory():
	# return the root folder, or '' if we could not find it
	# get the folder this file lives in
	try:
		filepath = os.path.dirname(os.path.realpath(__file__))
		# the root folder is 2 folders above this
		filepath = os.path.join(filepath, '../')
		filepath = os.path.join(filepath, '../')
		return os.path.normpath(filepath)
	except:
		return ''


def getDataDirectory(directory):
	root = getRootDirectory()
	if root == '':
		return root
	data = os.path.join(root, 'data')
	return os.path.join(data, directory)
	

def getAllFiles(folder, extension='wav'):
	# return a list of all the files in the WAV directory
	all_files = []
	for subdir, dirs, files in os.walk(folder):
		for file in files:
			filepath = subdir + os.sep + file
			if filepath.endswith('.{0}'.format(extension)):
				all_files.append(filepath)
	return all_files


def clearDirectory(path):
	# clear this directory
	try:
		shutil.rmtree(path)
	except:
		# ignore directories that do not exist
		pass
	# and then restore it
	os.mkdir(path)


def sliceAsWav(wav_file, output_folder, index, time=TIMESLICE):
	sound = AudioSegment.from_file(wav_file, format='wav')
	duration = len(sound)
	total_slices = (len(sound) - TIMESLICE) // TIME_DELTA
	for i in range(total_slices):
		start_time = i * TIME_DELTA
		new_slice = sound[start_time:start_time + TIMESLICE]
		new_slice.export('{0}/{1}.wav'.format(output_folder, getStringName(index)), format='wav')
		index += 1
	return index


def getStringName(index):
	number = str(index)
	return '{0}{1}'.format('0' * (6 - len(number)), number)


def convertWavFormat(filename, output_dir, frequency, channels, bytes_per_value, index):
	sound = AudioSegment.from_file(filename, format='wav')
	sound = sound.set_frame_rate(frequency)
	sound = sound.set_channels(channels)
	sound = sound.set_sample_width(bytes_per_value)
	sound.export('{0}/{1}.raw'.format(output_dir, getStringName(index)), format="raw")
