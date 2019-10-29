'''
Created on Dec 7, 2018

@author: jportegi1
'''
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.shrec import utils


class Experiment(object):
    '''
    classdocs
    '''

    # initial path ?
    # manage csv file / database

    def __init__(self, vae, experiment_params, train_data, path):
        '''
        Constructor
        '''
        self.vae_class = vae
        self.experiment_params = experiment_params
        self.train_data = train_data
        self.path = path
        self.csv_record = os.path.join(self.path, "diffusion_vae_experiments.csv")
        self.time_stamp = None

    def run(self, validation_data=None):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.vae_class.train_vae(self.train_data,
                                 self.experiment_params.epochs,
                                 self.experiment_params.batch_size,
                                 weights_file,
                                 tensorboard_file, validation_data=validation_data)
        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        vae_params_df = self.vae_class.params.params_to_df()
        vae_params_df.insert(0, "timestamp", self.time_stamp)
        vae_params_df["type"] = self.vae_class.type
        merged_df = pd.merge(vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_checkpoints(self):
        '''
        Runs experiment
        '''

        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Checkpoints folder
        weights_dir_checkpoint = os.path.join(weights_dir, self.time_stamp)
        os.makedirs(weights_dir_checkpoint, exist_ok=True)

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        self.vae_class.train_vae_checkpoints(self.train_data,
                                             self.experiment_params.epochs,
                                             self.experiment_params.batch_size,
                                             weights_file,
                                             tensorboard_file, weights_dir_checkpoint)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.vae_class.diffusion_vae_params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["manifold"] = self.vae_class.manifold
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_generator(self, low_freq_generator, steps_per_epoch):
        '''
        Runs experiment
        '''
        generator = low_freq_generator.generate()
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)

        # Fourier_components_file
        components_dir = os.path.join(self.path, "fourier_components")
        os.makedirs(components_dir, exist_ok=True)
        components_file = os.path.join(components_dir, self.time_stamp + '.npy')
        np.save(components_file, low_freq_generator._fourier_components)

        self.vae_class.train_generator_vae(generator, steps_per_epoch,
                                           self.experiment_params.epochs,
                                           weights_file,
                                           tensorboard_file)

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df = self.vae_class.diffusion_vae_params.params_to_df()
        diffusion_vae_params_df.insert(0, "timestamp", self.time_stamp)
        diffusion_vae_params_df["type"] = self.vae_class.type
        merged_df = pd.merge(diffusion_vae_params_df, experiment_params_df, on="timestamp")
        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def run_image_generator(self, generator, validation_generator = None, steps_per_epoch = None):
        '''
        Runs experiment
        '''
        self.time_stamp = time.strftime("%Y-%m-%d-%H-%M_")
        # Weight file
        weights_dir = os.path.join(self.path, "weights_folder")
        os.makedirs(weights_dir, exist_ok=True)
        weights_file = os.path.join(weights_dir, self.time_stamp + ".h5")

        # Tensorboard_file
        tensorboard_dir = os.path.join(self.path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        tensorboard_file = os.path.join(tensorboard_dir, self.time_stamp)
        self.vae_class.train_generator_vae(generator,
                                           epochs = self.experiment_params.epochs,
                                           weights=weights_file,
                                           tensorboard_file=tensorboard_file, validation_generator = validation_generator,steps_per_epoch = steps_per_epoch )

        # Append record of experiments to csv file
        experiment_params_df = self.experiment_params.params_to_df()
        experiment_params_df.insert(0, "timestamp", self.time_stamp)
        vae_params_df = self.vae_class.params.params_to_df()
        vae_params_df.insert(0, "timestamp", self.time_stamp)
        vae_params_df["type"] = self.vae_class.type
        merged_df = pd.merge(vae_params_df, experiment_params_df, on="timestamp")

        if os.path.isfile(self.csv_record):
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=False, index=False)
        else:
            with open(self.csv_record, 'a') as f:
                merged_df.to_csv(f, header=True, index=False)

    def plot_reconstructions(self):
        image_dir = os.path.join(self.path, "images")
        os.makedirs(image_dir, exist_ok=True)
        num_samples = 100
        z = np.random.normal(size=(num_samples, self.vae_class.latent_dim))
        for class_num in range(self.vae_class.num_classes):
            y = utils.one_hot_vectors(num_samples, self.vae_class.num_classes, class_num)
            decoded = self.vae_class.decode(z, y, 128)
            fig = plt.figure(figsize=(5, 5))
            size = int(np.sqrt(decoded.shape[1]))
            for i in range(num_samples):
                ax = fig.add_subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), i + 1)
                if self.vae_class.params.type_layers== "convolutional":
                    ax.imshow(decoded[i][:,:,0])
                else:
                    ax.imshow(decoded[i].reshape(size, size))
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(os.path.join(image_dir,str(self.time_stamp)+'_'+str(class_num)+"_reconstruction_"+'.png'))

    def plot_reconstructions(self):
        image_dir = os.path.join(self.path, "images")
        images = self.train_data[0]

        os.makedirs(image_dir, exist_ok=True)
        num_samples = 100
        z = np.random.normal(size=(num_samples, self.vae_class.latent_dim))
        for class_num in range(self.vae_class.num_classes):
            y = utils.one_hot_vectors(num_samples, self.vae_class.num_classes, class_num)
            decoded = self.vae_class.decode(z, y, 128)
            fig = plt.figure(figsize=(5, 5))
            size = int(np.sqrt(decoded.shape[1]))
            for i in range(num_samples):
                ax = fig.add_subplot(int(np.sqrt(num_samples)), int(np.sqrt(num_samples)), i + 1)
                if self.vae_class.params.type_layers== "convolutional":
                    ax.imshow(decoded[i][:,:,0])
                else:
                    ax.imshow(decoded[i].reshape(size, size))
                ax.set_xticks([])
                ax.set_yticks([])
            plt.savefig(os.path.join(image_dir,str(self.time_stamp)+'_'+str(class_num)+"_reconstruction_"+'.png'))


