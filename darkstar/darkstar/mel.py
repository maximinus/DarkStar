#!/usr/bin/env python3

import matplotlib
 # don't display any images
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display
import numpy as np
from helpers import *


MEL_WIDTH = 640
MEL_HEIGHT = 480
MEL_DPI = 80
MEL_WIDTH_INCHES = MEL_WIDTH // MEL_DPI
MEL_HEIGHT_INCHES = MEL_HEIGHT // MEL_DPI


def sliceAndConvert(root_folder, mel_folder):
	print('* Root: {0}'.format(root_folder))
	print('* Dest: {0}'.format(mel_folder))
	tmp_folder = getDataDirectory(TMP_FOLDER)
	file_count = 0
	# clear the output directory
	clearDirectory(mel_folder)
	# get the actual WAV files themselves
	for i in tqdm(getAllFiles(root_folder)):
		print('  * Converting {0} to MEL format time slices'.format(i.split('/')[-1]))
		clearDirectory(tmp_folder)
		sliceAsWav(i, tmp_folder)
		for j in tqdm(getAllFiles(tmp_folder)):
			# convert each WAV to a MEL
			filename = '{0}/{1}.png'.format(mel_folder, getStringName(file_count))
			createMelImage(j, filename)
			# update the output filename
			file_count += 1


def createMelImage(path, save_path):
	# take the WAV file at path and export as a MEL image in save_path
	sig, fs = librosa.load(path)
	pylab.rcParams['figure.figsize'] = 4, 3
	pylab.axis('off')
	# remove white edge
	pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
	spectrum = librosa.feature.melspectrogram(y=sig, sr=fs)
	librosa.display.specshow(librosa.power_to_db(spectrum, ref=np.max))
	pylab.savefig(save_path, bbox_inches=None, pad_inches=0, dpi=80)
	pylab.close()


def wavToMel(sound_file, save_path, xsize=MEL_WIDTH, ysize=MEL_HEIGHT):
	sig, fs = librosa.load(sound_file)
	pylab.rcParams['figure.figsize'] = MEL_WIDTH_INCHES, MEL_HEIGHT_INCHES
	pylab.axis('off')
	# remove white edge
	pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
	spectrum = librosa.feature.melspectrogram(y=sig, sr=fs)
	librosa.display.specshow(librosa.power_to_db(spectrum, ref=np.max))
	pylab.savefig(save_path, bbox_inches=None, pad_inches=0, dpi=MEL_DPI)
	pylab.close()

def melToWav(time_length):
	pass


if __name__ == '__main__':
	wav_file = getDataDirectory('TMP') + '/input.wav'
	mel_file = getDataDirectory('TMP') + '/spectrograph_highres.png'
	wavToMel(wav_file, mel_file)
