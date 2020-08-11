###############################################################################
#
# File: generic_unflow_input.py
#
# Functions as a generic directory input for the UnFlow implementation
#
# History:
# 07-30-20 - Levi Burner - Created file
#
###############################################################################

import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from motion_illusions.UnFlow.src.e2eflow.core.input import read_png_image, Input
import motion_illusions.utils.flow_plot as flow_plot

# This should likely be a resolution used to train otherwise strange errors can occur
# Chairs 512x384
# Kitti 1152x320, 768x320
# Synthia 768x512
# Cityscapes 1024x512
def get_data_resolution(image_dir):
    file_list = os.listdir(image_dir)
    file_list.sort()
    im = cv2.imread(os.path.join(image_dir, file_list[0]))
    print('input resolution: {}'.format(im.shape[:2]))
    return im.shape[:2]

class GenericUnflowInput(Input):
    def __init__(self, data, batch_size, *, num_threads=1, normalize=True, image_dir=None):
        dims = get_data_resolution(image_dir)
        super().__init__(data, batch_size, dims, num_threads=num_threads, normalize=normalize)
        self._image_dir = image_dir

    def input(self):
        file_list = os.listdir(self._image_dir)
        file_list.sort()

        full_file_names = [os.path.join(self._image_dir, f) for f in file_list]

        images_1 = read_png_image(full_file_names[:-1], 1)
        images_preprocessed_1 = self._preprocess_image(images_1)

        images_2 = read_png_image(full_file_names[1:], 1)
        images_preprocessed_2 = self._preprocess_image(images_2)

        input_shape = tf.shape(images_1)

        return tf.train.batch(
           [images_preprocessed_1, images_preprocessed_2, input_shape],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)
