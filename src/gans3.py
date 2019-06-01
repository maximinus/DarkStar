#!/usr/bin/env python3

# Adapation of code from the Neural-Nebula example

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
import os
import argparse
import glob

from PIL import Image
import matplotlib.pyplot as plt

import sys
import numpy as np
from tqdm import tqdm

from helpers import getAllFiles, getDataDirectory

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

class DCGAN():
    def __init__(self, img_cols=128, img_rows=128, channels=4, latent_dim=3, loss='binary_crossentropy', name='hirise'):
        self.name = name

        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_cols, self.img_rows, self.channels)
        print('* Image shape: {0}'.format(self.img_shape))
        self.latent_dim = latent_dim
        self.loss = loss
        self.optimizer = Adam(0.0005, 0.6)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        # Build the generator
        self.generator = self.build_generator()
        # Build the GAN
        self.build_combined()

    def build_combined(self):
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.loss, optimizer=self.optimizer)

    def build_generator(self):
        model = Sequential()
        #model.add(Dropout(0.1))
        #model.add(Dense(128, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        # This generator needs to have a 320x240x3 output
        model.add(Dense(64 * 32 * 32, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        model.add(Reshape((32, 32, 64)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same", activation="sigmoid", name="generator_output"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img, name="generator")

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        discrim = Model(img, validity)

        discrim.compile(loss='binary_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy'])
        return discrim

    def train(self, X_train, epochs, batch_size=128, save_interval=100):
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            #  Train Discriminator
            # select some random images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator
            # This always returns zero
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch % 10 == 0:
                print ("%d [D loss: %f, acc .: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs('{0}_{1}.png'.format(self.name, str(epoch)))
                # self.combined.save_weights("combined_weights ({}).h5".format(self.name)) # https://github.com/keras-team/keras/issues/10949
                self.generator.save_weights("generator ({}).h5".format(self.name))

    def save_imgs(self, name=''):
        # rows, columns to draw
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        # replace the first two latent variables with known values
        for i in range(r):
            for j in range(c):
                noise[4*i+j][0] = i/(r-1)-0.5
                noise[4*i+j][1] = j/(c-1)-0.5

        gen_imgs = self.generator.predict(noise)
        fig, axs = plt.subplots(r, c, figsize=(10,10))
        plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95, wspace=0.2, hspace=0.2)

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1

        filename = '{0}/{1}'.format(getDataDirectory('OUTPUT'), name)
        fig.savefig(filename, facecolor='black')
        plt.close()
        # ensure all memory is freed to avoid memory leaks
        plt.close('all')

def loadImages(images, score):

    # test: just load 500 files for now
    images = images[:5000]

    print('* Loading #{0} images for {1}'.format(len(images), score))
    x_train = []
    for i in tqdm(images):
        img = Image.open(i)
        # resize image to 128 * 128
        img = img.resize((128, 128), Image.LANCZOS) # resizes image in-place
        data = np.array(list(img.getdata())).reshape((img.size[0], img.size[1], -1))
        # remove the alpha channel, which is the last array of the array
        data = data[:,:,:3]
        # and normalize
        data = data.astype(np.float64)
        data /= 255
        x_train.append(data)
    return x_train

def create_dataset():
    # the size of all our images is 320x240
    # first let's get all of the images
    gd_data = loadImages(getAllFiles(getDataDirectory('MEL/GRATEFUL_DEAD'), 'png'), 1.0)
    x_train = np.array(gd_data)
    return x_train

if __name__ == '__main__':
    x_train = create_dataset()
    print('* Training shape: {0}'.format(x_train[0].shape))
    # make sure everything is ok
    assert(x_train.shape[0] > 0)
    dcgan = DCGAN(img_cols = x_train[0].shape[0],
                  img_rows = x_train[0].shape[1],
                  channels = x_train[0].shape[2],
                  latent_dim=32,
                  name='grateful_dead')
    dcgan.train(x_train, epochs=1000, batch_size=32, save_interval=50)
