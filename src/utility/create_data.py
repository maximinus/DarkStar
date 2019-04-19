#!/usr/bin/env python3

import numpy
from tqdm import tqdm

# take MEL images and create the final data file to input

def loadAsArray(files, value):
	data = []
	dtype = numpy.dtype('b')
	for i in tqdm(files):
		raw_array = numpy.fromfile(i, dtype)
		data.append([converted, value])
	return data


def saveData(all_data):
	random.shuffle(all_data)
	
	all_data = all_data[:200]
	
	x = numpy.array([i[0] for i in all_data])
	y = numpy.array([i[1] for i in all_data])
	print('* Learning shape:', x.shape)
	print('* Question shape:', y.shape)
	# save these arrays
	numpy.save('X_DATA', x)
	numpy.save('Y_DATA', y)	


def convertInput():
	print('Building data input files')
	files = getAllFiles('./MEL/GRATEFUL_DEAD')
	data_yes = loadAsArray(files, 1.0)
	files = getAllFiles('./MEL/OTHER')
	data_no = loadAsArray(files, 0.0)
	all_data = data_yes + data_no
	print('* Size of all data: {0}'.format(len(all_data)))
	saveData(all_data)


if __name__ == '__main__':
	convertInput()
