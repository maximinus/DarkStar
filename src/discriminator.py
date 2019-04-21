#!/usr/bin/env python3

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers, optimizers
import numpy as np

from helpers import getDataDirectory, getAllFiles

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
BATCH_SIZE = 16
EPOCHS = 100

def getSimpleModel():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Conv2D(64, (3,3)))
	model.add(Activation('relu'))
	# convert to 1d
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model


def getComplexModel():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(10, activation='softmax'))
	model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
	return model


def getDatagen():
	train_datagen = ImageDataGenerator()
	source = getDataDirectory('DATA/Train')

	train_generator = train_datagen.flow_from_directory(source, 
		target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')
	# repeat for testing
	test_datgen = ImageDataGenerator()
	source = getDataDirectory('DATA/Valid')
	test_generator = test_datgen.flow_from_directory(source,
		target_size=(IMAGE_WIDTH, IMAGE_HEIGHT), batch_size=BATCH_SIZE, class_mode='binary')
	return train_generator, test_generator


if __name__ == '__main__':
	train, test = getDatagen()
	model = getSimpleModel()
	training_size = len(getAllFiles(getDataDirectory('DATA/Train'), 'png'))
	validation_size = len(getAllFiles(getDataDirectory('DATA/Valid'), 'png'))
	model.fit_generator(generator=train,
    	                steps_per_epoch=2000 // BATCH_SIZE,
        	            validation_data=test,
            	        validation_steps=500 // BATCH_SIZE,
                	    epochs=EPOCHS)
