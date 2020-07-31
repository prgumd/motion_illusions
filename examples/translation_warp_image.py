###############################################################################
#
# File: translation_warp_image.py
#
# Continuously warp an image by a small translation
#
# History:
# 07-30-20 - Levi Burner - Created file
#
###############################################################################

import argparse
import time

import cv2
import numpy as np

import motion_illusions.utils.flow_plot as flow_plot
from motion_illusions.utils.image_tile import ImageTile
from motion_illusions.utils.signal_plot import SignalPlot
from motion_illusions.utils.rate_limit import RateLimit
from motion_illusions.utils.time_iterator import TimeIterator

from motion_illusions import rotation_translation_image_warp as warp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_path', help='image to warp and display')
    args = parser.parse_args()

    if args.image_path is None:
        raise ValueError('Program must be passed an image to warp')

    image = cv2.imread(args.image_path)

    session_name = 'translation_warp_image'
    tiler = ImageTile.get_instance(session=session_name)

    flow_vis_image = cv2.cvtColor(flow_plot.flow_direction_image(shape=image.shape), cv2.COLOR_HSV2BGR)

    # The visualization will be limited to this wall clock speed in hz
    rate_limit = RateLimit(limit_hz=60)

    last_wall_t = time.time()
    last_sim_t = time.time()

    # The simulation will be limited to this virtual speed in hz
    for sim_t in iter(TimeIterator(sim_rate_hz=60)):
        sim_delta_t = sim_t-last_sim_t
        wall_t = time.time()
        wall_delta_t = wall_t - last_wall_t

        # Generate a random translation from center
        std_dev_pixels = 5

        uv = np.random.normal(loc=0.0, scale=std_dev_pixels, size=(2,))

        optical_flow_trans = warp.discrete_optical_flow_due_to_2D_translation(
                                uv[0], uv[1], image.shape)
        warped_image = warp.image_warp(image, optical_flow_trans)

        tiler.add_image(image)
        tiler.add_image(warped_image)

        optical_flow_rot_image = cv2.cvtColor(flow_plot.visualize_optical_flow(optical_flow_trans), cv2.COLOR_HSV2BGR)
        tiler.add_image(optical_flow_rot_image)

        tiler.add_image(flow_vis_image)

        cv2.imshow(session_name, tiler.compose())
        cv2.setWindowTitle(session_name, session_name + ' real fps: {:.1f} sim fps: {:.1f}'.format(
            1.0 / wall_delta_t,
            1.0 / sim_delta_t))
        tiler.clear_scene()
        cv2.waitKey(1)

        rate_limit.sleep()
        last_sim_t = sim_t
        last_wall_t = wall_t
