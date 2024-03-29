"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
import datetime

########################################################
#import matplotlib.pyplot as plt
# newly added for saving plots in server
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
#######################################################

import sys
from src.data_loader_1picture import DataLoader
import numpy as np
import os

import keras.backend as K


class SRGAN():
    def __init__(self, dataset_name, upscale_power_factor, n_residual_blocks, local_path=None):
        # Input shape
        self.channels = 1
        self.hr_height = 512  # High resolution height
        self.hr_width = 512  # High resolution width
        self.checkpoint_path = 'checkpoints/'

        assert isinstance(upscale_power_factor, int), "upscale power factor must be int!"
        self.upscale_power_factor = upscale_power_factor
        self.upscale_factor = 2**self.upscale_power_factor
        self.lr_height = int(self.hr_height/self.upscale_factor)                 # Low resolution height
        self.lr_width = int(self.hr_width/self.upscale_factor)                  # Low resolution width

        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = n_residual_blocks #16

        self.optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        # self.vgg = self.build_vgg()
        # self.vgg.trainable = False
        # self.vgg.compile(loss='mse',
        #     optimizer=self.optimizer,
        #     metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = dataset_name  #'img_sst'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width),
                                      local_path=local_path)

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=self.optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)
        # Extract image features of the generated img
        # fake_hr_temp = Concatenate(axis=-1)([fake_hr, fake_hr])
        # fake_hr_temp = Concatenate(axis=-1)([fake_hr_temp, fake_hr])
        # fake_hr_temp = K.tile(fake_hr, (1, 1, 1, 3))
        # fake_features = self.vgg(fake_hr_temp)
        fake_features = Flatten()(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=self.optimizer)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        #img = Input(shape=self.hr_shape)
        img = Input(shape=(self.hr_height, self.hr_width, 3))
        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u = c2
        for upx in range(self.upscale_power_factor):
            u = deconv2d(u)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, sample_rslt_dir, batch_size=1, sample_interval=1, save_interval=1, load_checkpoint=False, checkpoint_id=0):

        start_time = datetime.datetime.now()

        N = 50
        start = 0

        if load_checkpoint:
            print('Models Loading ...')
            self.discriminator = load_model(self.checkpoint_path + str(checkpoint_id) + '-discriminator.h5')
            self.generator = load_model(self.checkpoint_path + str(checkpoint_id) + '-generator.h5')
            # High res. and low res. images
            img_hr = Input(shape=self.hr_shape)
            img_lr = Input(shape=self.lr_shape)

            # Generate high res. version from low res.
            fake_hr = self.generator(img_lr)
            # Extract image features of the generated img
            # fake_hr_temp = Concatenate(axis=-1)([fake_hr, fake_hr])
            # fake_hr_temp = Concatenate(axis=-1)([fake_hr_temp, fake_hr])
            # fake_hr_temp = K.tile(fake_hr, (1, 1, 1, 3))
            # fake_features = self.vgg(fake_hr_temp)
            fake_features = Flatten()(fake_hr)
            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # Discriminator determines validity of generated high res. images
            validity = self.discriminator(fake_hr)

            self.combined = Model([img_lr, img_hr], [validity, fake_features])
            self.combined.compile(loss=['binary_crossentropy', 'mse'],
                                  loss_weights=[1e-3, 1],
                                  optimizer=self.optimizer)

            start = checkpoint_id + 1

            print('Models Loaded Successfully! ')

        for epoch in range(start, epochs):

            for b_id in range(int(N / batch_size) + 1):
		# ----------------------
		#  Train Discriminator
		# ----------------------

		# Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
                imgs_hr = np.expand_dims(imgs_hr, axis=3)
                imgs_lr = np.expand_dims(imgs_lr, axis=3)
		# From low res. image generate high res. version
                fake_hr = self.generator.predict(imgs_lr)

                valid = np.ones((batch_size,) + self.disc_patch)
                fake = np.zeros((batch_size,) + self.disc_patch)

		# Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# ------------------
		#  Train Generator
		# ------------------

		# Sample images and their conditioning counterparts
                imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
                imgs_hr = np.expand_dims(imgs_hr, axis=3)
                imgs_lr = np.expand_dims(imgs_lr, axis=3)
		# The generators want the discriminators to label the generated images as real
                valid = np.ones((batch_size,) + self.disc_patch)
		# print(imgs_hr)
		# Extract ground truth image features using pre-trained VGG19 model
                # imgs_hr_temp = np.repeat(imgs_hr, 3, axis=-1)
                # image_features = self.vgg.predict(imgs_hr_temp)
                #print(imgs_hr.shape)
                image_features = imgs_hr.reshape(imgs_hr.shape[0], -1)
                #print(image_features.shape)
		# Train the generators
                g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

                elapsed_time = datetime.datetime.now() - start_time
		# Plot the progress
                print("epoch %d batch %d  time: %s" % (epoch, b_id, elapsed_time))

            if epoch % sample_interval == 0:
                self.sample_images(epoch, sample_rslt_dir)

            if epoch % save_interval == 0:
                print('Saving Models ...')
                self.discriminator.save(self.checkpoint_path + str(epoch) + '-discriminator.h5')
                self.generator.save(self.checkpoint_path + str(epoch) + '-generator.h5')
                #self.combined.save(self.checkpoint_path + str(epoch) + '-combined.h5')
                print('Models Saved Successfully!')

    def sample_images(self, epoch, sample_rslt_dir):
        if not os.path.exists(sample_rslt_dir):
            os.makedirs(sample_rslt_dir)
        #os.makedirs('images/%s' % self.dataset_dir, exist_ok=True)
        n_rows, n_cols = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=n_rows, is_testing=True)
        # vmin = np.zeros((n_rows,))
        # vmax = np.zeros((n_rows,))
        # for i in range(n_rows):
        #     vmin[i] = imgs_hr[i].min()
        #     vmax[i] = imgs_hr[i].max()

        imgs_hr = np.expand_dims(imgs_hr, axis=3)
        imgs_lr = np.expand_dims(imgs_lr, axis=3)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(n_rows, n_cols)
        cnt = 0
        for row in range(n_rows):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(np.squeeze(image[0], axis=2))
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig(sample_rslt_dir + "/{}.png".format(epoch))
        #fig.savefig("images/%s/%d.png" % (self.dataset_dir, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(n_rows):
            fig = plt.figure()
            plt.imshow(np.squeeze(imgs_lr[0], axis=2))
            fig.savefig(sample_rslt_dir + "/{}_lowers_{}.png".format(epoch, i))
            #fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_dir, epoch, i))
            plt.close()


