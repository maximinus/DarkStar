#!/usr/bin/env python3

import matplotlib
 # don't display any images
matplotlib.use('Agg')
import pylab
import librosa
import librosa.display
import numpy as np


MEL_WIDTH = 320
MEL_HEIGHT = 240


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
