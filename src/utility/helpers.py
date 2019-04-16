#!/usr/bin/env python3

import os
import shutil
from pydub import AudioSegment

from constants import TIMESLICE


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
	shutil.rmtree(path)
	# and then restore it
	os.mkdir(path)


def sliceAsWav(wav_file, output_folder):
	file_index = 0
	print('  > Slicing {0} as WAV'.format(wav_file))
	sound = AudioSegment.from_file(wav_file, format='wav')
	duration = len(sound)
	total_slices = len(sound) // TIMESLICE
	for i in range(total_slices):
		new_slice = sound[i * TIMESLICE: (i + 1) * TIMESLICE]
		print('{0}/{1}.wav'.format(output_folder, getStringName(file_index)))
		#new_slice.export('{1}/{2}.wav'.format(output_folder, self.getFilename('wav')), format='wav')
		file_index += 1


def getStringName(index):
	number = str(index)
	return '{0}{1}'.format('0' * (4 - len(number)), number)


def sliceFiles(root_folder, destination_folder):
	print('* Slicing files in {0} to {1}'.format(root_folder, destination_folder))
	clearDirectory(destination_folder)
	# we need to recreate all the files in the root
	for i in [f.path for f in os.scandir(root_folder) if f.is_dir()]:
		folder_count = 0
		for j in getAllFiles(i):
			# get the subdir folder name
			root = i.split('/')[-1]
			# add it to the destination folder
			dest_folder = os.path.join(destination_folder, root)
			# if it does not exist, create it
			try:
				os.makedirs(dest_folder)
			except:
				pass
			sliceAsWav(j, dest_folder)
			folder_count += 1
