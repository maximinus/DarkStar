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

Start by clearing out the slices directory. Now copy all the files you want, in the correct folder structure, to this directory.



There is a command line tool _convert_audio.py_ in the /commands folder. Simply run it with one argument - the location of the directory containing all the files.

The command will find all wav and mp3 files in this folder (or sub-folders) and normalise them:

 ::

     ./normalise.py ../music/{name-of-directory}

|
