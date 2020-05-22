#!/usr/bin/env python3

import os
import sys
import argparse
from tqdm import tqdm
from pydub import AudioSegment

"""
convert_audio {options} {input} {output}
"""

INPUT_FILES = ['wav']
BYTES_PER_VALUE = 2
BITRATE = 16384

def logError(text, exit=True):
	print(f'  Error: {text}')
	if exit == True:
		sys.exit()

def getArguments():
	parser = argparse.ArgumentParser(description='Convert audio')
	parser.add_argument('--bitrate', type=int, default=BITRATE, help='Output bitrate')
	parser.add_argument('--stereo', action='store_true', help='Output stereo')
	parser.add_argument('--split', type=int, default=0, help='Split into timed files')
	parser.add_argument('--recursive', action='store_true', help='Recurse through input')
	parser.add_argument('infolder', nargs='?', type=str, default='', help='Input folder')
	parser.add_argument('outfolder', nargs='?', type=str, default='', help='Output folder')
	args = parser.parse_args()
	return(args)

def checkArguments(args):
	if len(args.infolder) == 0:
		logError('No input specified')
	if len(args.outfolder) == 0:
		logError('No output folder specified')

def getFiles(args):
	# folder or file exists?
	if not os.path.exists(args.infolder):
		logError('Input path {args.infile} does not exist')
	if os.path.isfile(args.infolder):
		# must end with the right extension
		if args.infile[-3:].lower() in INPUT_FILES:
			return [os.path.dirname(args.infolder), args.infolder]
		logError('Input file is not a valid format')
	print('  Extracting files from {0}'.format(args.infolder))
	# must be a directory: are we recursive or not?
	files = []
	if args.recursive == False:
		# loop through all files and return them
		files = [x for x in os.listdir(args.infolder) if os.path.isfile(os.path.join(args.infolder, x))]
	else:
		print('All files')
		# recurse to get all files
		for root, dnames, fnames in os.walk(args.infolder):
			for f in fnames:
				files.append(os.path.join(root, f))
	# remove files we don'r care about
	files = [x for x in files if x[-3:].lower() in INPUT_FILES]
	files.insert(0, args.infolder)
	return files

def addFolderIfRequired(filepath):
	folder = os.path.dirname(filepath)
	if os.path.exists(folder):
		# check not a file
		if not os.path.isdir(folder):
			logError(f'Output folder {folder} is a file')
		return
	os.mkdir(folder)

def convertPaths(files, args):
	# the first file is the input dir, so remove that
	input_dir = files.pop(0)
	# we have a list of files for input
	# calculate the output for each of these (it'll be the output folder)
	if os.path.exists(args.outfolder):
		# make sure it is not a file
		if os.path.isfile(args.outfolder):
			logError('Output path cannot be a file')
	else:
		logError('Output folder does not exist')
	# calculate the output path
	paths = []
	normal_path = os.path.normpath(args.outfolder)
	for i in files:
		# we need to calculate the output file
		# we need to remove the matching path from the input file
		write_path = '{0}/{1}.raw'.format(normal_path, i[len(input_dir):-4])
		# convert extension to raw
		addFolderIfRequired(write_path)
		paths.append([i, write_path])
	return paths

def convertToRaw(input_file, output_file, bitrate, stereo=False):
	sound = AudioSegment.from_file(input_file, format='wav')
	sound = sound.set_frame_rate(bitrate)
	if stereo == True:
		sound = sound.set_channels(2)
		print('  Outputting as stereo')
	else:
		sound = sound.set_channels(1)
		print('  Outputting as mono')
	sound = sound.set_sample_width(BYTES_PER_VALUE)
	sound.export(output_file, format='raw')

def sliceToRaw(input_file, output_file, bitrate, stereo=False):
	# convert time to milliseconds
	time = args.split * 1000
	sound = AudioSegment.from_file(input_file, format='wav')
	sound = sound.set_frame_rate(bitrate)
	if stereo == True:
		sound = sound.set_channels(2)
	else:
		sound = sound.set_channels(1)
	sound = sound.set_sample_width(BYTES_PER_VALUE)
	duration = len(sound)
	total_slices = len(sound) // time
	for i in range(total_slices):
		start_time = i * time
		new_slice = sound[start_time:start_time + time]
		extra_name = 't{0}t{1}'.format(start_time // 1000, (start_time + time) // 1000)
		new_name = output_file[:-4] + '_{0}.raw'.format(extra_name)
		new_slice.export(new_name, format='raw')

def convertSingleFile(i, args):
	input_file = i[0]
	output_file = i[1]
	if args.split == 0:
		# dont split the files, so fairly easy
		convertToRaw(i[0], i[1], args.bitrate, args.stereo)
	else:
		# we need to do the same but split up first
		sliceToRaw(i[0], i[1], args.bitrate, args.stereo)

def convertFiles(files, args):
	for i in tqdm(files):
		convertSingleFile(i, args)

def test():
	time = 4 * 1000
	sound = AudioSegment.from_file('test.wav', format='wav')
	sound = sound.set_frame_rate(16384)
	sound = sound.set_channels(1)
	sound = sound.set_sample_width(BYTES_PER_VALUE)
	duration = len(sound)
	start_time = 0 * time
	new_slice = sound[start_time:start_time + time]
	new_slice.export('result.raw', format='wav')

if __name__ == '__main__':
	args = getArguments()
	checkArguments(args)
	# now start the process
	files = getFiles(args)
	files = convertPaths(files, args)
	convertFiles(files, args)
