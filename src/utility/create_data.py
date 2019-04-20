#!/usr/bin/env python3

import shutil
import random
from tqdm import tqdm
from helpers import getAllFiles, getDataDirectory, clearDirectory

# take MEL images and sort to output directories


# move a list of files from a to b
def copyFiles(files, destination):
	print('* Copying files to {0}'.format(destination))
	for i in tqdm(files):
		# copy, don't move the files
		shutil.copy(i, destination)


def setupMelFiles():
	# clear directories
	for i in ['DATA/Train/GD', 'DATA/Valid/GD', 'DATA/Train/Other', 'DATA/Valid/Other']:
		clearDirectory(getDataDirectory(i))

	# get the GD files
	gd = getAllFiles(getDataDirectory('MEL/GRATEFUL_DEAD'), 'png')
	sample_size = int(len(gd) * 0.8)
	# shuffle in place
	random.shuffle(gd)
	gd_train = gd[:sample_size]
	gd_valid = gd[sample_size:]
	copyFiles(gd_train, getDataDirectory('DATA/Train/GD'))
	copyFiles(gd_valid, getDataDirectory('DATA/Valid/GD'))

	# same with the others
	other = getAllFiles(getDataDirectory('MEL/OTHER'), 'png')
	sample_size = int(len(other) * 0.8)
	random.shuffle(other)
	other_train = other[:sample_size]
	other_valid = other[sample_size:]
	copyFiles(other_train, getDataDirectory('DATA/Train/Other'))
	copyFiles(other_valid, getDataDirectory('DATA/Valid/Other'))


if __name__ == '__main__':
	setupMelFiles()
