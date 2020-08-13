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

import glob
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

from motion_illusions.UnFlow.src.e2eflow.core.input import read_png_image, Input
import motion_illusions.utils.flow_plot as flow_plot

# Copied straight from UnFlow/src/e2eflow/middlebury/input.py
def _read_flow(filenames, num_epochs=None):
    """Given a list of filenames, constructs a reader op for ground truth flow files."""
    filename_queue = tf.train.string_input_producer(filenames,
        shuffle=False, capacity=len(filenames), num_epochs=num_epochs)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    value = tf.reshape(value, [1])
    value_width = tf.substr(value, 4, 4)
    value_height = tf.substr(value, 8, 4)
    width = tf.reshape(tf.decode_raw(value_width, out_type=tf.int32), [])
    height = tf.reshape(tf.decode_raw(value_height, out_type=tf.int32), [])

    value_flow = tf.substr(value, 12, 8 * width * height)
    flow = tf.decode_raw(value_flow, out_type=tf.float32)
    flow = tf.reshape(flow, [height, width, 2])
    mask = tf.to_float(tf.logical_and(flow[:, :, 0] < 1e9, flow[:, :, 1] < 1e9))
    mask = tf.reshape(mask, [height, width, 1])

    return flow, mask

# This should likely be a resolution used to train otherwise strange errors can occur
# Chairs 512x384
# Kitti 1152x320, 768x320
# Synthia 768x512
# Cityscapes 1024x512
def get_data_resolution(image_dir):
    file_list = glob.glob(os.path.join(image_dir, 'image_*'))
    file_list.sort()
    im = cv2.imread(file_list[0])
    print('input resolution: {}'.format(im.shape[:2]))
    return im.shape[:2]

class GenericUnflowInput(Input):
    def __init__(self, data, batch_size, *, num_threads=1, normalize=True, image_dir=None):
        dims = get_data_resolution(image_dir)
        super().__init__(data, batch_size, dims, num_threads=num_threads, normalize=normalize)
        self._image_dir = image_dir

    # Copied straight from UnFlow/src/e2eflow/middlebury/input.py
    def _preprocess_flow(self, t, channels):
        height, width = self.dims
        # Reshape to tell tensorflow we know the size statically
        return tf.reshape(self._resize_crop_or_pad(t), [height, width, channels])

    def input(self):
        # Read in images
        image_file_list = glob.glob(os.path.join(self._image_dir, 'image_*'))
        image_file_list.sort()

        images_1 = read_png_image(image_file_list[:-1], 1)
        images_preprocessed_1 = self._preprocess_image(images_1)

        images_2 = read_png_image(image_file_list[1:], 1)
        images_preprocessed_2 = self._preprocess_image(images_2)

        input_shape = tf.shape(images_1)

        # Read in flow
        flow_file_list = glob.glob(os.path.join(self._image_dir, 'flow_*'))
        flow_file_list.sort()
        flow_file_list = flow_file_list[1:]

        flow, mask = _read_flow(flow_file_list)
        flow = self._preprocess_flow(flow, 2)
        mask = self._preprocess_flow(mask, 1)

        return tf.train.batch(
           [images_preprocessed_1, images_preprocessed_2, input_shape,
            flow, mask],
           batch_size=self.batch_size,
           num_threads=self.num_threads,
           allow_smaller_final_batch=True)
