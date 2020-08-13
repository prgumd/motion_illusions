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

    for r in result:
        # TODO what order is im1 and im2
        # Convert all flow returns to be appropriate
        # Masks for the flow validity

        (image, im2, error, flow, flow_gt) = r
        tiler.add_image(cv2.cvtColor(image[0, :, :, :], cv2.COLOR_RGB2BGR))
        tiler.add_image(cv2.cvtColor(im2[0, :, :, :], cv2.COLOR_RGB2BGR))
        tiler.add_image(cv2.cvtColor(error[0, :, :, :], cv2.COLOR_RGB2BGR))

        image = cv2.cvtColor(image[0, :, :, :], cv2.COLOR_RGB2BGR)

        flow = flow[0, :, :, :]
        flow_gt = flow_gt[0, :, :, :]
        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(flow, np.copy(image), color=(255, 0, 0))
        tiler.add_image(optical_flow_rot_image)

        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(flow_gt, np.copy(image), color=(0, 255, 0))
        tiler.add_image(optical_flow_rot_image)

        flow_diff = flow - flow_gt
        # tiler.add_image(cv2.cvtColor(flow_diff[0, :, :, :], cv2.COLOR_RGB2BGR))
        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(flow_diff, np.copy(image), quiver_scale=10.0, color=(0, 0, 255))
        tiler.add_image(optical_flow_rot_image)


        cv2.imshow(session_name, tiler.compose())
        tiler.clear_scene()
        cv2.waitKey()
