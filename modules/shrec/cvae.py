#!/usr/bin/anaconda3/bin/python3
# Consistency with previous versions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# System imports
import os
import sys

sys.path.append(os.getcwd())

# Tensorflow and keras
import tensorflow as tf
from keras.layers import Lambda, Input, Concatenate
from keras.models import Model
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras import callbacks
from keras import backend as K
from keras.optimizers import Adam
from modules.shrec import utils

# Plot and numpy
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')  # for plotting in the cluster


class CVAE():
    def __init__(self, encoder_x,encoder_xy, decoder, cvae_params):

        self.params = cvae_params

        self.image_size = cvae_params.image_size
        self.image_shape = self.params.image_shape
        self.num_classes = self.params.num_classes
        self.intermediate_dim = self.params.intermediate_dim

        self.latent_dim = self.params.latent_dim  # latent dimension depends on manifold

        self.var_x = cvae_params.var_x  # decoding distribution variance (normal distribution)
        self.r_loss = cvae_params.r_loss
        self.encoder_x = encoder_x
        self.encoder_xy = encoder_xy
        self.decoder = decoder
        self.vae_train, self.kl_loss, self.r_loss, self.mean_squared_error, self.vae_loss, self.classification_loss  = self.build_vae_train(self.encoder_x, self.encoder_xy, self.decoder)
        self.vae_test = self.build_vae_test(self.encoder_x, self.encoder_xy, self.decoder)





    def build_vae_train(self, encoder_x, encoder_xy, decoder):
        x = Input(shape=self.image_shape, name="vae_train_x_input")
        y = Input(shape=(self.num_classes,), name="vae_train_y_input")

        # Calculate distribution parameters
        z_log_var, pi_cat = encoder_x(x)
        mu_z = encoder_xy([x,y])

        # Sample latent variable
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([mu_z, z_log_var])
        zy = Concatenate(axis=-1)([z, y])
        mu_x = decoder(zy)
        cvae_train = Model([x, y], mu_x)
        kl_loss, r_loss, mean_squared_error, vae_loss, classification_loss = self.build_loss_functions(z_log_var,
                                                                                                       mu_z, pi_cat, x,
                                                                                                       y)
        optimizer = Adam(lr = self.params.learning_rate)
        cvae_train.compile(optimizer=optimizer, loss=vae_loss,
                           metrics=[r_loss, kl_loss, mean_squared_error, classification_loss])
        return cvae_train, kl_loss, r_loss, mean_squared_error, vae_loss, classification_loss

    def build_vae_test(self, encoder_x, encoder_xy, decoder):
        x = Input(shape=self.image_shape, name="vae_train_x_input")

        z_log_sigma, pi_cat = encoder_x(x)

        mu_z = encoder_xy([x, pi_cat])
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([mu_z, z_log_sigma])
        zpi = Concatenate(axis=-1)([z, pi_cat])
        mu_x = decoder(zpi)
        cvae_test = Model(x, mu_x)
        return cvae_test

    def sampling(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.

        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)

        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        z_sample = z_mean + tf.multiply(epsilon, tf.exp(0.5 * z_log_var))
        return z_sample

    def build_loss_functions(self, z_log_var, z_mean, pi_cat, x, y):
        def classification_loss(inputs, outputs):
            loss = categorical_crossentropy(y, pi_cat)
            return loss

        #def classification_loss(inputs, outputs):
        #    loss = 1-tf.cos(pi_cat-y)
        #    return loss

        def kl_loss(inputs, outputs):
            """
            Kullback-Leibler divergence of posterior distribution. No necessary inputs and outputs are needed
            :param inputs :
            :param outputs:
            :return:
            """
            loss = 0.5 * tf.reduce_sum(K.exp(z_log_var) + K.square(z_mean) - z_log_var - 1, axis=-1)
            return loss

        def r_loss(inputs, outputs):
            """
            Reconstruction loss part of the variational autoencoder
            :param inputs: input data
            :param outputs: output data from the autoencoder
            :return: r_loss tensor
            """
            if self.r_loss == "mse":
                print("Reconstruction loss is mean squared error")
                if self.params.type_layers == "convolutional":
                    se = K.sum(K.pow(outputs - x, 2), axis=-1)
                    se = K.sum(se, axis=-1)
                    se = K.sum(se, axis=-1)
                elif self.params.type_layers == "dense":
                    se = K.sum(K.pow(outputs - x, 2), axis=-1)
                loss = 0.5 * (se / self.var_x + self.image_size * np.log(2 * np.pi * self.var_x))
                loss = 0.5 * se
            elif self.r_loss == "binary":
                print("Reconstruction loss is binary cross entropy")
                epsilon = K.epsilon()
                loss = x * tf.log(epsilon + outputs) \
                       + (1 - x) * tf.log(epsilon + 1 - outputs)
                loss = -tf.reduce_sum(loss, axis=-1)
                if self.params.type_layers == "convolutional":
                    loss = tf.reduce_sum(loss, axis = -1)
                    loss = tf.reduce_sum(loss, axis=-1)
            else:
                print("Error, no renconstruction chosen")
                loss = None
            return loss

        def mean_squared_error(inputs, outputs):
            """
            Calculates the mean squared error between input data and output data
            :param inputs:
            :param outputs:
            :return: r_loss tensor
            """
            calculated_mse = mse(x, outputs)
            return calculated_mse

        def vae_loss(inputs, outputs):
            loss = K.mean(r_loss(inputs, outputs) + kl_loss(inputs, outputs) + classification_loss(inputs, outputs))
            return loss

        return kl_loss, r_loss, mean_squared_error, vae_loss, classification_loss

    def train_vae(self, train_data, epochs, batch_size, weights_file, tensorboard_file, validation_data = None):
        """

        :param train_data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of each the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :param tensorboard_file (str) complete path for saving the tensorboard log
        :return:
        """
        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae_train.fit(train_data, train_data[0],
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=[tensorboard_cb],
                           verbose=2, validation_data=validation_data
                           )
        self.vae_train.save_weights(weights_file)

    def train_vae_checkpoints(self, input_data, epochs, batch_size, weights_file, tensorboard_file, models_filepath):
        """
        Train diffusion variational autoencoder that can
        :param train_data (numpy array): first dimension corresponds to number of datapoints
        while second dimension corresponds to the size of the datapoint
        :param epochs (int) number of epochs the diffusion vae is trained
        :param batch_size (int) size of the batch used for training each epoch
        :param weights_file (str) complete path for saving the trained weights
        :param tensorboard_file (str) complete path for saving the tensorboard log
        :param models_filepath (str) path to where the diffusion vae models are to be saved
        :return:
        """
        checkpoint = callbacks.ModelCheckpoint(models_filepath, verbose=0, save_best_only=False,
                                               save_weights_only=True, mode='auto', period=10)

        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        self.vae_train.fit(input_data, input_data[0],
                           epochs=epochs,
                           batch_size=batch_size,
                           callbacks=[tensorboard_cb, checkpoint],
                           verbose=2
                           )
        self.vae_train.save_weights(weights_file)

    def train_generator_vae(self, generator, epochs, weights, tensorboard_file, validation_generator = None, steps_per_epoch = None):
        """
        Train the diffusion vae whose data is generated from a generator
        :param generator (generator): generator object that can be used with fit_generator
        :param steps_per_epoch:
        :param epochs:
        :param weights:
        :param tensorboard_file:
        :return:
        """
        tensorboard_cb = callbacks.TensorBoard(log_dir=tensorboard_file)
        if validation_generator!=None:
            early_stop = callbacks.EarlyStopping(monitor="classification_loss", patience = 20)
            self.vae_train.fit_generator(generator, epochs = epochs, verbose=2, callbacks=[tensorboard_cb, early_stop],
                                         workers=1,
                                         use_multiprocessing=False,
                                         validation_data= validation_generator)
        else:
            self.vae_train.fit_generator(generator, epochs=epochs, verbose=2, callbacks=[tensorboard_cb],
                                         workers=1,
                                         use_multiprocessing=False,
                                         validation_data=validation_generator)

        self.vae_train.save_weights(weights)


    def load_model(self, weight_file):
        """
        Reload the weights of previously trained models
        :param weight_file (str): path to the stored weights file
        :return:
        """
        self.vae_train.load_weights(weight_file)

    def encode(self, x, batch_size):
        """
        Encode into the latent space the input data
        :param data (numpy array) first dimension of array corresponds to the number of
        datapoints and the second correspond to the size of each datapoint
        :param batch_size (int):
        :return:
        """

        log_var_z, pi_cat = self.encoder_x.predict(x, batch_size=batch_size)
        mu_z = self.encoder_xy.predict([x, pi_cat], batch_size=batch_size)
        return mu_z, log_var_z, pi_cat

    def encode_time(self, data, batch_size):
        time = np.exp(self.encoder.predict(data, batch_size=batch_size)[1])
        return time

    def decode(self, z, y, batch_size):
        decoded = self.decoder.predict([z,y], batch_size=batch_size)
        return decoded
