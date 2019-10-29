#!/usr/bin/anaconda3/bin/python3
# Consistency with previous versions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# System imports
import os
import sys

sys.path.append(os.getcwd())


from keras.layers import Input, Dense, Conv2D, Flatten, Concatenate, BatchNormalization, UpSampling2D, MaxPool2D, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
from modules.shrec.cvae import CVAE


class CVAE_Basic(CVAE):
    def __init__(self, params):
        self.type = "basic"
        self.params = params
        self.latent_dim = self.params.latent_dim  # latent dimension depends on manifold
        self.image_shape =self.params.image_shape
        self.intermediate_dim = self.params.intermediate_dim
        self.num_classes = self.params.num_classes


        encoder_x = self.build_encoder_x()
        encoder_xy = self.build_encoder_xy()
        decoder = self.build_decoder()
        super(CVAE_Basic, self).__init__(encoder_x, encoder_xy, decoder, params)



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
        inputs_x = Input(shape=self.image_shape, name='encoder_x_input')
        # Initialize the list of layers
        encoder_x_layers = [inputs_x]
        for layer in range(self.params.num_encoding_layers):

            # Convolutional network
            if self.params.type_layers == "convolutional":
                encoder_x_layers.append(Conv2D(filters=self.intermediate_dim * 2 ** layer,
                                               kernel_size=self.params.kernel_size,
                                               strides=1,
                                               padding='same',
                                               activation='relu',
                                               name="h_enc_c_" + str(layer))(encoder_x_layers[-1]))
                if self.params.double_convolution:
                    encoder_x_layers.append(Conv2D(filters=self.intermediate_dim * 2 ** layer,
                                                   kernel_size=self.params.kernel_size,
                                                   strides=1,
                                                   padding='same',
                                                   activation='relu',
                                                   name="h_enc_c2_" + str(layer))(encoder_x_layers[-1]))
                encoder_x_layers.append(MaxPool2D(pool_size = (2,2), name = "h_max_pool_"+str(layer))(encoder_x_layers[-1]))
                #encoder_x_layers.append(BatchNormalization()(encoder_x_layers[-1]))

            # Dense network
            elif self.params.type_layers == "dense":
                encoder_x_layers.append(Dense(self.intermediate_dim,
                                              activation='relu',
                                              name="h_enc_x_d_" + str(layer))(encoder_x_layers[-1]))

        if self.params.type_layers == "convolutional":
            self.intermediate_conv_shape = K.int_shape(encoder_x_layers[-1])
            encoder_x_layers.append(Flatten()(encoder_x_layers[-1]))

        # Determine the scale and the category
        z_log_var = Dense(self.latent_dim, name="z_log_var")(encoder_x_layers[-1])
        pi_cat = Dense(self.num_classes, name="pi_cat", activation='softmax')(encoder_x_layers[-1])

        # Build the encoder
        encoder_x = Model(inputs_x, [z_log_var, pi_cat], name="encoder_on_x")
        return encoder_x

    def build_encoder_xy(self):
        ################################################################################################################
        # ENCODER XY
        ################################################################################################################
        # Define the encoding layers that depend on both x and y
        inputs_x = Input(shape=self.image_shape, name='encoder_xy_xinput')
        inputs_y = Input(shape=(self.num_classes,), name ='encoder_xy_y_input')
        if self.params.type_layers =="convolutional":
            x_flat = Flatten()(inputs_x)
        else:
            x_flat = inputs_x
        inputs_xy = Concatenate(axis = -1)([x_flat, inputs_y])
        encoder_xy_layers = [inputs_xy]
        for layer in range(self.params.num_encoding_layers):
            if self.params.type_layers == "convolutional":
                encoder_xy_layers.append(Dense(self.intermediate_dim*(2**self.params.num_encoding_layers)*self.params.kernel_size**2,
                                               activation='relu',
                                               name="h_enc_xy_d_" + str(layer))(encoder_xy_layers[-1]))
            elif self.params.type_layers == "dense":
                encoder_xy_layers.append(Dense(self.intermediate_dim,
                                          activation='relu',
                                          name="h_enc_xy_d_" + str(layer))(encoder_xy_layers[-1]))
        z_mean = Dense(self.latent_dim, name='z_mean')(encoder_xy_layers[-1])
        encoder_xy = Model([inputs_x, inputs_y], z_mean, name='encoder_on_xy')
        return encoder_xy

    def build_decoder(self):
        input_zy = Input(shape=(self.latent_dim + self.num_classes,), name="decoder_input")
        decoder_layers = [input_zy]
        if self.params.type_layers == "convolutional":
            decoder_layers.append(Dense(np.product(self.intermediate_conv_shape[1:]))(decoder_layers[-1]))
            decoder_layers.append(Reshape((self.intermediate_conv_shape[1],
                                           self.intermediate_conv_shape[2],
                                           self.intermediate_conv_shape[3]))(decoder_layers[-1]))

        for layer in range(self.params.num_decoding_layers-1):
            if self.params.type_layers == "convolutional":
                  decoder_layers.append(UpSampling2D(size=(2, 2), data_format=None, name="upsample_"+str(layer))(decoder_layers[-1]))
                  decoder_layers.append(Conv2D(filters=self.intermediate_dim * 2 ** (
                             self.params.num_decoding_layers - layer - 1),
                                               kernel_size=self.params.kernel_size,
                                               strides=1,
                                               padding='same',
                                               activation='relu',
                                               name="h_dec_c_" + str(layer))(decoder_layers[-1]))
            elif self.params.type_layers == "dense":
                decoder_layers.append(Dense(self.intermediate_dim,
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
            print("Convolutional")
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


