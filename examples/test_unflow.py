###############################################################################
#
# File: test_unflow.py
#
# Test the installation of the unflow implementation
#
# History:
# 07-30-20 - Levi Burner - Created file
#
###############################################################################

import argparse
import time
import os
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

import motion_illusions.utils.flow_plot as flow_plot
from motion_illusions.utils.image_tile import ImageTile
from motion_illusions.generic_unflow_input import GenericUnflowInput
from motion_illusions.evaluate_unflow import evaluate_experiment
from motion_illusions.UnFlow.src.e2eflow.util import config_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', dest='image_dir', help='folder to get images from')
    args = parser.parse_args()

    if args.image_dir is None:
        raise ValueError('Program must be passed a directory to load images from')

    unflow_input = GenericUnflowInput(data=None, batch_size=1, normalize=False, image_dir=args.image_dir)

    print('Running Unflow')
    current_config = config_dict('./unflow_config.ini')
    result, image_names = evaluate_experiment(current_config['dirs'],
                                              'CSS', unflow_input, 15)

    session_name = 'rotation_warp_image'
    tiler = ImageTile.get_instance(session=session_name)

    for (image, error, flow) in result:
        tiler.add_image(cv2.cvtColor(image[0, :, :, :], cv2.COLOR_RGB2BGR))
        tiler.add_image(cv2.cvtColor(error[0, :, :, :], cv2.COLOR_RGB2BGR))
        tiler.add_image(cv2.cvtColor(flow[0, :, :, :], cv2.COLOR_RGB2BGR))

        cv2.imshow(session_name, tiler.compose())
        tiler.clear_scene()
        cv2.waitKey()
