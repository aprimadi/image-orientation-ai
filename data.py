from __future__ import print_function
import MySQLdb
import keras
from keras.preprocessing.image import img_to_array, apply_transform
from keras.preprocessing.image import transform_matrix_offset_center
from keras.utils.data_utils import Sequence
import numpy as np
import cv2
import random

class DataGenerator:
    def __init__(self, data):
        self.data = data

    def flow(self, batch_size=32, shuffle=True, seed=None):
        return NumpyArrayIterator(
            data=self.data,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.index_array = None
        self.total_batches_seen = 0

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError(
                'Asked to retrieve element {idx}, but the Sequence has length' '{length}'.format(idx=idx, length=len(self))
            )
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array_from = self.batch_size * idx
        index_array_to = self.batch_size * (idx + 1)
        index_array = self.index_array[index_array_from:index_array_to]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size    # round up

    def on_epoch_end(self):
        self._set_index_array()

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        raise NotImplementedError

class NumpyArrayIterator(Iterator):
    def __init__(self, data, batch_size, shuffle, seed):
        self.data = data
        self.width_shift_range = 0.1
        self.height_shift_range = 0.1
        super(NumpyArrayIterator, self).__init__(len(data), batch_size, shuffle, seed)

    def _read_image(self, image_id):
        path = "data-sanitized/%07d.png" % image_id
        img = cv2.imread(path)
        img = img_to_array(img)
        img.astype('float32')
        img /= 255
        return img

    def _get_batches_of_transformed_samples(self, index_array):
        img_row_axis = 0
        img_col_axis = 1
        img_channel_axis = 2

        batch_x = np.zeros(tuple([len(index_array)] + [128, 128, 3]),
                           dtype='float32')
        batch_y = np.zeros((len(index_array), 4), dtype='float32')
        for i, j in enumerate(index_array):
            image_id = self.data[j][0]
            rotation = self.data[j][1]
            x = self._read_image(image_id)

            transform_matrix = None

            # Randomly rotate images
            theta = random.choice([0, 90, 180, 270])
            thetar = np.deg2rad(theta)
            rotation_matrix = np.array([[np.cos(thetar), -np.sin(thetar), 0],
                                        [np.sin(thetar), np.cos(thetar), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
            rotation = (rotation + int(theta / 90)) % 4

            if self.height_shift_range:
                tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
                if self.height_shift_range < 1:
                    tx *= x.shape[img_row_axis]
            else:
                tx = 0

            if self.width_shift_range:
                ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
                if self.width_shift_range < 1:
                    ty *= x.shape[img_col_axis]
            else:
                ty = 0

            if tx != 0 or ty != 0:
                shift_matrix = np.array([[1, 0, tx],
                                         [0, 1, ty],
                                         [0, 0, 1]])
                if transform_matrix is None:
                    transform_matrix = shift_matrix
                else:
                    transform_matrix = np.dot(transform_matrix, shift_matrix)

            # Apply transform
            x = apply_transform(x, transform_matrix, img_channel_axis, fill_mode='nearest', cval=0.)

            y = np.zeros(4)
            y[rotation] = 1
            batch_x[i] = x
            batch_y[i] = y
        return batch_x, batch_y
