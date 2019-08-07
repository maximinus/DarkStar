#!/usr/bin/env python3

import json
from datetime import datetime
from keras.callbacks import Callback
from darkstar.helpers import *


class ReportData:
	def __init__(self):
		stamp = datetime.now().strftime('%d-%b-%y-%H:%M:%S')
		directory = getDataDirectory('REPORTS')
		self.filename = '{0}/{1}.json'.format(directory, stamp)
		print('-> Using filename: {0}'.format(self.filename))
		self.epochs = []

	def addEpoch(self, epoch, logs):
		# epoch is what number we are on
		# logs is a dictionary of data
		self.epochs.append(logs)
		self.saveData()

	def saveData(self):
		with open(self.filename, 'w') as json_file:
			json.dump(self.epochs, json_file, indent=4)


class Recorder(Callback):
	def __init__(self):
		super().__init__()
		# get date / time and create filename
		self.data = ReportData()

	def on_epoch_end(self, epoch, logs={}):
		#  update batch info
		self.data.addEpoch(epoch, logs)
