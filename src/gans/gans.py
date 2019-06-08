#!/usr/bin/env python3

import os
import sys
import glob
import argparse

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

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../')
from helpers import getDataDirectory, getAllFiles


EPOCHS = 2500

class DCGAN():
    def __init__(self, img_rows=128, img_cols=128, channels=4, latent_dim=3, loss='binary_crossentropy', name='hirise'):
        # rows  = 240, cols = 320 always for the current test
        self.name = name

        # Input shape
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
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
        #model.add(Dense(128, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        #model.add(Dropout(0.1))

        model.add(Dense(64 * 40 * 30, activation="relu", input_dim=self.latent_dim, name="generator_input") )
        model.add(Reshape((30, 40, 64)))
        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Dropout(0.1))
        model.add(UpSampling2D())

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
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
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
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch % 5 == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                folder = getDataDirectory('OUTPUT')
                self.save_imgs('{}/{}_{:05d}.png'.format(folder, self.name, epoch))


    def save_imgs(self, name=''):
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

        if name:
            fig.savefig(name, facecolor='black' )
        else:
            fig.savefig('{}.png'.format(self.name), facecolor='black' )
        plt.close()


def create_dataset(x_size=320, y_size=240, directory='dataset/'):
    # this puts all the images into one massive dataset and returns it
    images = getAllFiles(directory, 'png')
    x_train = []
    y_train = []
    for i in tqdm(range(200)):
        img = Image.open(images[i])
        x_train.append(np.array(list(img.getdata()), dtype=float).reshape((y_size, x_size, -1)))
        y_train.append([0,0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return (x_train, y_train)


if __name__ == '__main__':
    path = getDataDirectory('DATA/Train/GD')
    x_train, y_train = create_dataset(320, 240, directory=path)

    # turn training dataset into floats
    x_train /= 255
    dcgan = DCGAN(img_rows = x_train[0].shape[0],
                  img_cols = x_train[0].shape[1],
                  channels = x_train[0].shape[2],
                  latent_dim=32,
                  name='grateful_dead')
    dcgan.train(x_train, epochs=EPOCHS, batch_size=32, save_interval=25)
