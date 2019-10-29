#!/usr/bin/anaconda3/bin/python3
# Consistency with previous versions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# System imports
import os
import sys

sys.path.append(os.getcwd())

from keras.layers import Input, Dense, Conv2D, Flatten, Reshape, Concatenate, UpSampling2D, Dropout
from keras.models import Model

from modules.shrec.cvae import CVAE
from modules.shrec.external_module import vgg16_places_365
# Plot and numpy
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # for plotting in the cluster


class CVAE_VGG(CVAE):
    def __init__(self, params):

        self.params = params
        self.type = "VGG"

        self.latent_dim = self.params.latent_dim  # latent dimension depends on manifold
        self.vgg = self.build_vgg()
        encoder_x = self.build_encoder_x()
        encoder_xy = self.build_encoder_xy()
        decoder = self.build_decoder()
        super(CVAE_VGG, self).__init__(encoder_x, encoder_xy, decoder, params)
        # self.encoder, self.decoder, self.vae = self.build_network()


    def build_vgg(self):
        vgg_model = vgg16_places_365.VGG16_Places365(include_top = False, weights = 'places', input_tensor= None, input_shape = self.params.image_shape, pooling = None,  trainable = False)
        return vgg_model

    def build_encoder_x(self):
        """
        Build the encoder network that only depends on the input image, includes z_log_var and pi_cat (scale parameter
        of encoding distribution and category)
        :return: encoder_x model
        """
        ################################################################################################################
        # ENCODER X
        ################################################################################################################
        # Define the encoding layers that depend only on x
        inputs_x = Input(shape=self.params.image_shape, name='encoder_x_input')
        x = self.vgg(inputs_x)
        x = Flatten()(x)
        x = Dense(100, activation='relu', name="dense_x_1")(x)
        #x = Dropout(0.5)(x)
        x = Dense(100, activation='relu', name="dense_x_2")(x)
        #x = Dropout(0.5)(x)
        pi_cat = Dense(self.params.num_classes, activation="softmax", name="pi_cat")(x)
        z_log_var = Dense(self.params.latent_dim, name="z_log_var")(x)

        # Build the encoder
        encoder_x = Model(inputs_x, [z_log_var, pi_cat], name="encoder_on_x")
        return encoder_x

    def build_encoder_xy(self):
        ################################################################################################################
        # ENCODER XY
        ################################################################################################################
        # Define the encoding layers that depend on both x and y
        inputs_x = Input(shape=self.params.image_shape, name="encoder_xy_inputx")
        inputs_y = Input(shape=(self.params.num_classes,), name = "encoder_xy_inputy")
        vgg_x = self.vgg(inputs_x)
        flat_vgg_x = Flatten()(vgg_x)
        inputs_xy = Concatenate(axis = -1)([flat_vgg_x, inputs_y])
        z_mean = Dense(100, activation='relu', name = "dense_xy_1")(inputs_xy)
        #z_mean =  Dropout(0.5)(z_mean)
        z_mean = Dense(100, activation='relu', name="dense_xy_2")(z_mean)
        #z_mean = Dropout(0.5)(z_mean)
        z_mean = Dense(self.latent_dim, name='z_mean')(z_mean)
        encoder_xy = Model([inputs_x, inputs_y], z_mean, name='encoder_on_xy')
        return encoder_xy

    def build_decoder(self):
        input_zy = Input(shape=(self.params.latent_dim + self.params.num_classes,), name="decoder_input")
        decoder_layers = [input_zy]
        if self.params.type_layers == "convolutional":
            decoder_layers.append(Dense(int(np.product(np.array(self.params.image_shape[:2])/2**5)*512))(decoder_layers[-1]))
            decoder_layers.append(Reshape((int(self.params.image_shape[0]/2**5),
                                           int(self.params.image_shape[1]/2**5),
                                           512))(decoder_layers[-1]))

        for layer in range(self.params.num_decoding_layers-1):
            if self.params.type_layers == "convolutional":
                  decoder_layers.append(UpSampling2D(size=(2, 2), data_format=None, name="upsample_"+str(layer))(decoder_layers[-1]))
                  decoder_layers.append(Conv2D(filters=int(512 / 2 ** (layer+1)),
                                               kernel_size=self.params.kernel_size,
                                               strides=1,
                                               padding='same',
                                               activation='relu',
                                               name="h_dec_c_" + str(layer))(decoder_layers[-1]))
            elif self.params.type_layers == "dense":
                decoder_layers.append(Dense(self.params.intermediate_dim,
                                            activation='relu',
                                            name="h_dec_d_" + str(layer))(decoder_layers[-1]))

        if self.params.r_loss == 'mse':
            output_activation = 'linear'
        elif self.params.r_loss == 'binary':
            output_activation = 'sigmoid'
        else:
            print("Loss not appropriately chosen")
            output_activation = 'none'

        if self.params.type_layers == "convolutional":
            decoder_layers.append(UpSampling2D(size=(2, 2), data_format=None, name="upsample_" + str(layer+1))(decoder_layers[-1]))
            mu_x = Conv2D(filters=self.params.image_shape[-1],
                         kernel_size=self.params.kernel_size,
                         strides=1,
                         padding='same',
                         activation=output_activation,
                         name="output")(decoder_layers[-1])

        elif self.params.type_layers == "dense":
            mu_x = Dense(self.params.image_size,
                         activation=output_activation,
                         name="outputs")(decoder_layers[-1])
        else:
            mu_x = None
        decoder = Model(input_zy, mu_x)
        return decoder
