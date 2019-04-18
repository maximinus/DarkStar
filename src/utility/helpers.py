#!/usr/bin/env python3

import os
from tqdm import tqdm
import shutil
from pydub import AudioSegment

from mel import createMelImage
from constants import TIMESLICE, TIME_DELTA, TMP_FOLDER


def getRootDirectory():
	# return the root folder, or '' if we could not find it
	# get the folder this file lives in
	try:
		filepath = os.path.dirname(os.path.realpath(__file__))
		filepath = os.path.join(filepath, '../..')
		# the root folder is 2 folders above this
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


def sliceAsWav(wav_file, output_folder):
	file_index = 0
	sound = AudioSegment.from_file(wav_file, format='wav')
	duration = len(sound)
	total_slices = (len(sound) - TIMESLICE) // TIME_DELTA
	for i in range(total_slices):
		start_time = i * TIME_DELTA
		new_slice = sound[start_time:start_time + TIMESLICE]
		new_slice.export('{0}/{1}.wav'.format(output_folder, getStringName(file_index)), format='wav')
		file_index += 1


def getStringName(index):
	number = str(index)
	return '{0}{1}'.format('0' * (4 - len(number)), number)


def sliceAndConvert(root_folder, mel_folder):
	print('* Root: {0}'.format(root_folder))
	print('* Dest: {0}'.format(mel_folder))
	tmp_folder = getDataDirectory(TMP_FOLDER)
	# 1000 files per folder
	folder_count = 0
	file_count = 0
	# clear the output directory
	clearDirectory(mel_folder)
	# create the fist directory
	current_output = '{0}/{1}'.format(mel_folder, getStringName(folder_count))
	clearDirectory(current_output)
	# get the actual WAV files themselves
	for i in tqdm(getAllFiles(root_folder)):
		print('  * Converting {0} to MEL format time slices'.format(i.split('/')[-1]))
		clearDirectory(tmp_folder)
		sliceAsWav(i, tmp_folder)
		for j in tqdm(getAllFiles(tmp_folder)):
			# convert each WAV to a MEL
			filename = '{0}/{1}.png'.format(current_output, getStringName(file_count))
			createMelImage(j, filename)
			# update the output filename
			file_count += 1
			if file_count > 999:
				file_count = 0
				folder_count += 1
				current_output = '{0}/{1}'.format(mel_folder, getStringName(folder_count))
				clearDirectory(current_output)
