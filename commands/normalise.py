#!/usr/bin/env python3

import os
import sys
import subprocess
from darkstar.helpers import getAllFiles

# normalize all the files in the WAV folders

def normalize(folders):
	files = []
	for i in folders:
		files.extend(getAllFiles(i))
	print(f'  Normalising {len(files)} files')
	files.insert(0, 'normalize-audio')
	subprocess.run(files)


if __name__ == '__main__':
	# pass with the folder you with to normalize
	if len(sys.argv) < 2:
		print('  Error: Please supply the folder name')
		sys.exit(False)
	for folder in sys.argv[1:]:
		if not os.path.isdir(folder):
			print(f'  Error: {folder} is not a folder')
			sys.exit(False)
		else:
			print(f'  Importing files from {folder}')
	normalize(sys.argv[1:])
	print('Files normalised')
