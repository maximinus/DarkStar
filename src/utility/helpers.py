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
		return filepath
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


class Slicer:
	# slicer has the job of taking an input directory and slicing all the files
	# to an output directory
	def __init__(self, files, directory):
		self.index = 0
		self.dir_index = 0
		self.files = files
		self.output = '{0}/data/SLICES/'
		clearDirectory(self.output)

	def getFilename(self, extension='raw'):
		number = str(self.index)
		self.index += 1
		return '{0}{1}.{2}'.format('0' * (6 - len(number)), number, extension)

	def getDirectoryName(self):
		number = str(self.dir_index)
		self.dir_index += 1
		return '{0}{1}'.format('0' * (4 - len(number)), number)

	def sliceAsWav(self):
		total = 0
		for filename in self.files:
			print('  > Slicing {0} as WAV'.format(filename))
			dir_name = self.getDirectoryName()
			os.mkdir('{0}/{1}'.format(self.output, dir_name))
			sound = AudioSegment.from_file(filename, format='wav')
			duration = len(sound)
			total_slices = len(sound) // TIMESLICE
			#for i in range(total_slices):
			#	new_slice = sound[i * TIMESLICE: (i + 1) * TIMESLICE]
			#	new_slice.export('{0}/{1}/{2}'.format(self.output, dir_name, self.getFilename('wav')), format='wav')
			#	total += 1
		return total


def sliceFiles(root_folder, destination_folder):
	print('* Slicing files in {0} to {1}'.format(root_folder, destination_folder))
	# we need to recreate all the files in the root
	for i in [f.path for f in os.scandir(root_folder) if f.is_dir()]:
		print('--> {0}'.format(i))
