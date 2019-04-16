#!/usr/bin/env python3

import os
import shutil
from pydub import AudioSegment

from constants import TIMESLICE


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
	def __init__(self, files, directory):
		self.index = 0
		self.dir_index = 0
		self.files = files
		self.output = './SLICES/{0}'.format(directory)
		clearDirectory(self.output)


	def getFilename(self, extension='raw'):
		number = str(self.index)
		self.index += 1
		return '{0}{1}.{2}'.format('0' * (6 - len(number)), number, extension)

	def getDirectoryName(self):
		number = str(self.dir_index)
		self.dir_index += 1
		return '{0}{1}'.format('0' * (4 - len(number)), number)


	def sliceFiles(self, format='raw'):
		total = 0
		for filename in self.files:
			print('  > Slicing {0}'.format(filename))
			dir_name = self.getDirectoryName()
			os.mkdir('{0}/{1}'.format(self.output, dir_name))
			sound = AudioSegment.from_file(filename, format='wav')
			duration = len(sound)
			total_slices = len(sound) // TIMESLICE
			for i in range(total_slices):
				new_slice = sound[i * TIMESLICE: (i + 1) * TIMESLICE]
				# convert and save this slice
				new_slice = new_slice.set_frame_rate(4096)
				new_slice = new_slice.set_channels(1)
				new_slice = new_slice.set_sample_width(1)
				new_slice.export('{0}/{1}/{2}'.format(self.output, dir_name, self.getFilename()), format=format)
				total += 1
		return total

	def sliceAsWav(self):
		total = 0
		for filename in self.files:
			print('  > Slicing {0} as WAV'.format(filename))
			dir_name = self.getDirectoryName()
			os.mkdir('{0}/{1}'.format(self.output, dir_name))
			sound = AudioSegment.from_file(filename, format='wav')
			duration = len(sound)
			total_slices = len(sound) // TIMESLICE
			for i in range(total_slices):
				new_slice = sound[i * TIMESLICE: (i + 1) * TIMESLICE]
				new_slice.export('{0}/{1}/{2}'.format(self.output, dir_name, self.getFilename('wav')), format='wav')
				total += 1
		return total
