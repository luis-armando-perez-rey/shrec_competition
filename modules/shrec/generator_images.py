
import numpy as np
import keras
import imageio
from modules.shrec import utils
import itertools
#import cv2
from PIL import Image


class ShrecGenerator(keras.utils.Sequence):

    def __init__(self, list_paths, labels, batch_size=32, dim=(64,64), n_channels=1,
                 n_classes=10, flatten = False, shuffle=True, reflect = False, resize = False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_paths = list_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.flatten = flatten
        self.shuffle = shuffle
        self.reflect = reflect
        self.resize = resize
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.reflect:
            return int(2*np.floor(len(self.list_paths) / self.batch_size))
        else:
            return int(np.floor(len(self.list_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]



        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_paths))
        if self.reflect:
            reflections = [True, False]
            self.combinations =  list(itertools.product(self.indexes, reflections))
            self.indexes = np.arange(len(self.combinations))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype = int)
        y = np.empty((self.batch_size, self.n_classes), dtype=int)


        # Generate data
        for i, index in enumerate(indexes):

            if self.reflect:
                datapoint_number = self.combinations[index][0]
            else:
                datapoint_number = index
            # Store sample
            try:
                image = imageio.imread(self.list_paths[datapoint_number])[:,:,:self.n_channels]
            except:
                image = imageio.imread(self.list_paths[datapoint_number])
                if image.ndim == 2:
                    # print("Anomalous shape {}".format(image.shape))
                    image_amplified = np.zeros((image.shape[0], image.shape[1], self.n_channels),
                                               dtype=type(image[0, 0]))
                    for i in range(self.n_channels):
                        image_amplified[:, :, i] = image
                    image = image_amplified
                print("Couldn't read image")
            #except:
            #    image = imageio.imread(self.list_paths[datapoint_number])[:, :,np.newaxis]

#            image = Image.open(self.list_paths[datapoint_number])


            # Resize if necessary
            #if self.resize:
            #    image = image.resize((self.dim[0], self.dim[1]), Image.ANTIALIAS)
#            image = np.array(image, dtype =np.float64)

            # Reflect in the necessary case
            if self.reflect and self.combinations[index][1]:
                image = np.flip(image, axis=1)


            X[i,] = image
            # Store class
            y[i] = self.labels[datapoint_number]
        if self.flatten:
            X = utils.flatten_normalize_images(X[:,:,:,0])
        else:
            X = utils.normalize_images(X)
        return [X, y], X



