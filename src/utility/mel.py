#!/usr/bin/env python3

import matplotlib
 # don't display any images
matplotlib.use('Agg')
import pylab
import numpy as np
import librosa
import librosa.display


def createMelFromAudio(path, output_dir):
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
    return(save_path)


def createAudioFromMel(path, output_dir):
    # TODO: Implement
    # (see https://github.com/librosa/librosa/pull/844/commits)
    pass


if __name__ == '__main__':
    EXAMPLE_WAV = '../tests/resources/WAV/mono_16_khz.wav'
    createMelFromAudio(EXAMPLE_WAV, '../tests/tmp')
