#!/usr/bin/env python3

import subprocess
from helpers import getDataDirectory, getAllFiles, clearDirectory

# normalize all the files in the WAV folders

if __name__ == '__main__':
	# get all the files
	files = getAllFiles(getDataDirectory('WAV'))
	print('* Normalising {0} files'.format(len(files)))
	files.insert(0, 'normalize-audio')
	subprocess.run(files)
