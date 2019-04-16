#!/usr/bin/env python3

import matplotlib
 # don't display any images
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display
import numpy as np

from helpers import clearDirectory, getAllFiles, Slicer
from constants import GRATEFUL_DEAD, MEL_FOLDER, SLICES

# find the files
# convert to some format
# convert to mel

def slice():
	print('  Slicing GD files...')
	wav_files = getAllFiles('./WAV/{0}'.format(GRATEFUL_DEAD))[:1]
	print('    (Found {0} GD files)'.format(len(wav_files)))
	foo = Slicer(wav_files, GRATEFUL_DEAD)
	return foo.sliceAsWav()


def createMelImage(path, output_dir):
	sig, fs = librosa.load(path)
	# make pictures name 
	save_path = './{0}/{1}'.format(output_dir, '{0}.png'.format(path.split('/')[-1].split('.')[0]))
	pylab.axis('off')
	# remove white edge
	pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
	spectrum = librosa.feature.melspectrogram(y=sig, sr=fs)
	librosa.display.specshow(librosa.power_to_db(spectrum, ref=np.max))
	pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
	pylab.close()
	print('    Saved MEL as {0}'.format(save_path))


def convertToMel(directory):
	# clear the mel directory
	output_dir = './{0}/{1}'.format(MEL_FOLDER, directory)
	source_dir = './{0}/{1}'.format(SLICES, directory)
	clearDirectory(output_dir)
	# get all the sliced file
	print('  Searching for files to convert in {0}'.format(source_dir))
	files = getAllFiles(source_dir)
	print('  Found {0} files to convert to MEL'.format(len(files)))
	for i in files:
		createMelImage(i, output_dir)
	print('  Saved {0} MEL files'.format(len(files)))


if __name__ == '__main__':
	print('  Sliced {0} files'.format(slice()))
	# now convert the slices to a MEL format
	convertToMel(GRATEFUL_DEAD)
