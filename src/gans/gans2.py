#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Dropout, Input
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from tqdm import tqdm


def loadData():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.astype(np.float32) - 127.5) / 127.5#
	# convert shape from (60k, 28, 28) to (60k, 784)
	x_train = x_train.reshape(60000, 784)
	return (x_train, y_train, x_test, y_test)


def adamOptimizer():
	return adam(lr=0.002, beta_1=0.5)


def createGenerator():
	generator = Sequential()
	generator.add(Dense(units=256, input_dim=100))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(units=512))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(units=1024))
	generator.add(LeakyReLU(0.2))

	generator.add(Dense(units=784, activation='tanh'))

	generator.compile(loss='binary_crossentropy', optimizer=adamOptimizer())
	return generator


def createDiscriminator():
	discriminator = Sequential()
	discriminator.add(Dense(units=1024, input_dim=784))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(Dropout(0.3))

	discriminator.add(Dense(units=512))
	discriminator.add(LeakyReLU(0.2))
	discriminator.add(Dropout(0.3))

	discriminator.add(Dense(units=256))
	discriminator.add(LeakyReLU(0.2))

	discriminator.add(Dense(units=1, activation='sigmoid'))
	discriminator.compile(loss='binary_crossentropy', optimizer=adamOptimizer())
	return discriminator


def createGan(discriminator, generator):
	discriminator.trainable = False
	gan_input = Input(shape=(100,))
	x = generator(gan_input)
	gan_output = discriminator(x)
	gan = Model(inputs=gan_input, outputs=gan_output)
	gan.compile(loss='binary_crossentropy', optimizer='adam')
	return gan


def plotImages(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
	noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
	images = generator.predict(noise)
	images = images.reshape(100, 28, 28)
	plt.figure(figsize=figsize)
	for i in range(images.shape[0]):
		plt.subplot(dim[0], dim[1], i+1)
		plt.imshow(images[i], interpolation='nearest')
		plt.axis('off')
	plt.tight_layout()
	plt.savefig('generated_{0}.png'.format(epoch))
	plt.close('all')



def training(epochs=1, batch_size=128):
	# load the data
	(x_train, y_train, x_test, y_test) = loadData()
	batch_count = x_train.shape[0] // batch_size

	# create the gan
	generator = createGenerator()
	discriminator = createDiscriminator()
	gan = createGan(discriminator, generator)

	for e in range(1, epochs + 1):
		for _ in tqdm(range(batch_size)):
			# generate noise
			noise = np.random.normal(0, 1, [batch_size, 100])
			images = generator.predict(noise)
			# and get real images
			image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]
			# put them together
			x = np.concatenate([image_batch, images])
			y_dis = np.zeros(2 * batch_size)
			y_dis[:batch_size] = 0.9

			discriminator.trainable = True
			discriminator.train_on_batch(x, y_dis)

			noise = np.random.normal(0, 1, [batch_size, 100])
			y_gen = np.ones(batch_size)

			discriminator.trainable = False
			gan.train_on_batch(noise, y_gen)

			if (e == 1) or (e % 10 == 0):
				plotImages(e, generator)


if __name__ == '__main__':
	training(100, 128)
