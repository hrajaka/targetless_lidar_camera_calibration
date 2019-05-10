import cv2
import math
import os
import re
from glob import glob

from keras.utils import Sequence

import numpy as np
import scipy.misc
from keras.preprocessing.image import load_img
from imgaug import augmenters as iaa


class DataSequence(Sequence):

    def __init__(self, data_dir, batch_size, image_shape):
        # from keras website to handle memory issue
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
        self.label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}
        self.sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        # augmentation for poor lightning condition
        self.aug_pipe = iaa.Sequential(
            [
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

    def __len__(self):
        return int(math.ceil(len(self.image_paths) / float(self.batch_size)))

    def get_batch_images(self, idx, path_list):
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            # load the image and resize
            image = load_img(im)
            image = scipy.misc.imresize(image, (self.image_shape[1], self.image_shape[0]))
            # augment the image
            image = self.aug_pipe.augment_image(image)
            return np.array([image])


    def get_batch_labels(self, idx, path_list):
        # iterate and map the mask labels for the respective images
        for im in path_list[idx * self.batch_size: (1 + idx) * self.batch_size]:
            gt_image_file = self.label_paths[os.path.basename(im)]
            gt_image = load_img(gt_image_file)
            gt_image = scipy.misc.imresize(gt_image, (self.image_shape[1], self.image_shape[0]))
            background_color = np.array([255, 0, 0])
            gt_bg = np.all(gt_image == background_color, axis=2)
            gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
            gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
            return np.array([gt_image])

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """
        batch_x = self.get_batch_images(idx, self.image_paths)
        batch_y = self.get_batch_labels(idx, self.image_paths)
        return batch_x, batch_y
