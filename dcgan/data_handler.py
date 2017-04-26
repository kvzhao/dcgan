import numpy as np
import h5py
import os, sys
import collections
from random import shuffle

DATASETNAME='SQUAREICE_STATES.h5'
IMAGE_SIZE=32

def read_data_sets(data_path,
                   reshape=False,
                   ):
    IMAGES = 'ICESTATES'
    h = h5py.File(data_path + DATASETNAME, 'r+')
    train_images = h[IMAGES][:]
    train_labels = np.ones_like(train_images)
    h.close()

    data = DataSet(images=train_images, labels=train_labels, grayscale=True)
    return data


class DataSet(object):
	def __init__ (self,
                      images,
                      labels,
                      dtype=np.float32,
                      grayscale=True,
                      reshape=False):
            self._images = images
            self._labels = labels
            self._num_of_samples = images.shape[0]
            self._epochs_completed = 0
            self._index_in_epoch = 0
            if(grayscale):
                self._convert_to_grayscale()

        @property
        def images(self):
            return self._images

        @property
        def labels(self):
            return self._labels

        @property
        def num_samples(self):
            return self._num_of_samples

        @property
        def epochs_completed(self):
            return self._epochs_completed

	def _convert_to_grayscale(self):
            for i in range(self._num_of_samples):
                c = self._images[i]
                c[c <= 0.0] = 0.0
                self._images[i] = c
            self._images = self._images.astype(np.uint8)
            #  And shuffl the data
            perm = np.arange(self._num_of_samples)
            np.random.shuffle(perm)
            self._images = self.images[perm]
            self._labels = self.labels[perm]

	def next_batch(self, batch_size, shuffle=True):
		start = self._index_in_epoch
		#self._index_in_epoch += batch_size

                # Shuffle the first epoch
                if self._epochs_completed == 0 and start == 0 and shuffle:
                    perm0 = np.arange(self._num_of_samples)
                    np.random.shuffle(perm0)
                    self._images = self.images[perm0]
                    self._labels = self.labels[perm0]
                # Finsh the epoch
                if start + batch_size > self._num_of_samples:
                    self._epochs_completed += 1
                    rest_num_samples = self._num_of_samples - start
                    images_rest_part = self._images[start:self._num_of_samples]
                    labels_rest_part = self._labels[start:self._num_of_samples]
                     # Shuffle
                    if shuffle:
                        perm = np.arange(self._num_of_samples)
                        np.random.shuffle(perm)
                        self._images = self.images[perm]
                        self._labels = self.labels[perm]
                    # Start next epoch
                    start = 0
                    self._index_in_epoch = batch_size - rest_num_samples
                    end = self._index_in_epoch
                    images_new_part = self._images[start:end]
                    labels_new_part = self._labels[start:end]
                    return np.concatenate((images_rest_part, images_new_part), axis=0), \
                            np.concatenate((labels_rest_part, labels_new_part), axis=0)
                else:
                    self._index_in_epoch += batch_size
                    end = self._index_in_epoch
                    return self._images[start:end], self._labels[start:end]


def read_datasets(data_path='./dataset/'):
    return read_data_sets(data_path)
