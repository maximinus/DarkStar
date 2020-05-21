Preparing Audio
===============

Ensure that all audio is stored under the /music folder (these files are not hosted on github).

Make sure that tracks such as "dead air", "audience", "tuning" and so forth are deleted.

They should all be 16-bit 44.1kHz WAV files.

Different experiments will need different audio files, but in general we normalise, slice and convert to raw audio file before training.

|

Converting Audio
----------------

The first step you will likely want to do is normalise all the files. Because this step is destructive, we will want to put the files somewhere else first.

Start by clearing out the slices directory. Now copy all the files you want, in the correct folder structure, to this directory - let's call this directory "normal".


The normalise.py command will find all wav and mp3 files in this folder (or sub-folders) and normalise them:

::

    ./normalise.py ../music/{name-of-directory}


The next part is to split the audio into slices.

There is a command line tool _convert_audio.py_ in the /commands folder. Simply run it with one argument - the location of the directory containing all the files.

We'll want to convert each of the different types on their own, as they will get placed in the same output directory. The output directory will also preserve the directory structure of the files

::

    ./convert_audio.py --bitrate 4096 --split 2 --recursive {input_folder} {output_folder}

The output folder will now contain our custom raw files ready for machine learning.

|
