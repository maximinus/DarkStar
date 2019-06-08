DarkStar
--------

Machine learning project to create new Grateful Dead shows

Aim:

* Train a GD discriminator
* Build a GANS to produce GD audio

## Starting Instructions

The first goal is to produce a discriminator that is able to tell the difference between GD and not GD.

This is done in several steps. First, let's get our setup done:

* git clone https://github.com/maximinus/DarkStar.git
* cd DarkStar
* python3 -m venv PYENV-FOLDER-NAME
* source PYENV-FOLDER-NAME/bin/activate
* pip install -r requirements.txt
* sudo apt-get install normalize-audio

Then we organise the basic audio. This step is quite quick.

* Download a bunch of SBD shows (5/6 is enough) and a bunch of other sound files to about the same length of time.
* Easiest way for this is with the youtube-dl script: ```youtube-dl -x --audio-format wav YOUTUBE-URL```
* Create a folder data in the root directory. inside that folder, create folders DATA, MEL, SLICES, TMP and WAV.
* In the this WAV folder, create a folder GRATEFUL_DEAD and another OTHER.
* Convert all of your music to WAV and place the GD files in GRATEFUL_DEAD (you may have subdirectories) and the rest in OTHER.
* Delete all files that are marked tuning / audience or similar in the GD folder.


Next we need to turn the audio into the correct format (in our case, 320 x 240 MEL spectrograms). This takes around 12 hours on my machine.

* First, we normalise the audio volume levels. Run the python script ```normalise.py```.
* This will take all the files from the WAV folders, normalize them and put them into the data/NORMALISE folder.
* Now we need to turn them into a series of 320 * 240 MEL spectrograms. ```Run convert.py```. THIS WILL TAKE SOME TIME!
* Finally, seperate the resultant MEL images into the required data structure. Run ```create_data.py```


Finally you can train the discriminator. Run ```discriminator.py```. This will also take some time. At the end, a file *grateful-dead.h5* will be created - this will be the trained neural net.

Currently I find that it generally does not take many epochs to be 90% accurate, but more testing is needed.


## For Training the GANS

* Add Tkinter: ```sudo apt-get install python3-tk```
* Ensure that all the steps above have been run for data preperation (everything up to running the discriminator).
* Create a folder OUTPUT in the data directory
* Run ```gans.py```. This will take some considerable time.
