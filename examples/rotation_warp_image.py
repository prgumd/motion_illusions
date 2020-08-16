###############################################################################
#
# File: rotation_warp_image.py
#
# Continuously warp an image by a small random rotation
#
# History:
# 07-28-20 - Levi Burner - Created file
#
###############################################################################

import argparse
import time
import os

import cv2
import numpy as np

import motion_illusions.utils.flow_plot as flow_plot
from motion_illusions.utils.image_tile import ImageTile
from motion_illusions.utils.signal_plot import SignalPlot
from motion_illusions.utils.rate_limit import RateLimit
from motion_illusions.utils.time_iterator import TimeIterator

from motion_illusions import rotation_translation_image_warp as warp
from motion_illusions import opencv_optical_flow

# Copied from UnFlow/src/eval_gui.py
def write_flo(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    height, width = flow.shape[:2]
    magic.tofile(f)
    np.int32(width).tofile(f)
    np.int32(height).tofile(f)
    data = np.float32(flow).flatten()
    data.tofile(f)
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', dest='image_path', help='image to warp and display')
    parser.add_argument('--save_dir', dest='save_dir', help='folder to save generated image sequence')
    args = parser.parse_args()

    if args.image_path is None:
        raise ValueError('Program must be passed an image to warp')

    image = cv2.imread(args.image_path)

    # Assume an HFOV of 5 degrees
    hfov_deg = 5.0
    focal_length = (image.shape[0] / 2) / np.tan((hfov_deg/2)*180.0/np.pi)

    session_name = 'rotation_warp_image'
    tiler = ImageTile.get_instance(session=session_name, max_width=768*3, scale_factor=1.0)

    flow_vis_image = cv2.cvtColor(flow_plot.flow_direction_image(shape=image.shape), cv2.COLOR_HSV2BGR)

    # The visualization will be limited to this wall clock speed in hz
    rate_limit = RateLimit(limit_hz=60)

    last_wall_t = time.time()
    last_sim_t = time.time()

    frame_id = 0
    optical_flow_rot_last = warp.discrete_optical_flow_due_to_rotation(0, 0, 0, focal_length, image.shape)
    last_warped_image = np.zeros(image.shape).astype(np.uint8)

    # The simulation will be limited to this virtual speed in hz
    for sim_t in iter(TimeIterator(sim_rate_hz=60)):
        sim_delta_t = sim_t-last_sim_t
        wall_t = time.time()
        wall_delta_t = wall_t - last_wall_t

        # Generate a random yaw, pitch, roll
        # It would be better to simulate a random walk on a sphere in a 4 dimensional space
        std_dev_deg = 0.2

        ypr = np.random.normal(loc=0.0, scale=std_dev_deg*np.pi/180.0, size=(3,))

        optical_flow_rot = warp.discrete_optical_flow_due_to_rotation(
                                ypr[0], ypr[1], 0,
                                focal_length, image.shape)
        warped_image = warp.image_warp(image, optical_flow_rot)
        intermediate_flow = optical_flow_rot - optical_flow_rot_last

        if args.save_dir:
            file_name = 'image_{:06d}.png'.format(frame_id)
            cv2.imwrite(os.path.join(args.save_dir, file_name), warped_image)

            flo_name = 'flow_{:06d}.flo'.format(frame_id)
            write_flo(intermediate_flow, os.path.join(args.save_dir, flo_name))
            frame_id += 1

        tiler.add_image(image)
        tiler.add_image(last_warped_image)
        tiler.add_image(warped_image)

        intermediate_warped = warp.image_warp(last_warped_image, intermediate_flow)
        tiler.add_image(intermediate_warped)

        tiler.add_image(np.abs(intermediate_warped.astype(np.float32) - warped_image.astype(np.float32)).astype(np.uint8))

        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(optical_flow_rot_last, np.copy(image), color=(255, 0, 0))
        tiler.add_image(optical_flow_rot_image)

        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(optical_flow_rot, np.copy(optical_flow_rot_image), color=(0, 255, 0))
        tiler.add_image(optical_flow_rot_image)

        optical_flow_rot_image = flow_plot.dense_flow_as_quiver_plot(intermediate_flow, np.copy(optical_flow_rot_image), color=(0, 0, 255))
        tiler.add_image(optical_flow_rot_image)

        # optical_flow_rot_image = cv2.cvtColor(flow_plot.visualize_optical_flow(optical_flow_rot), cv2.COLOR_HSV2BGR)
        # tiler.add_image(optical_flow_rot_image)

        #tiler.add_image(flow_vis_image)

        # lk_flow_list = opencv_optical_flow.lucas_kanade(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        #                                                 cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY))
        # lk_flow_subtracted = flow_plot.subtract_dense_flow_from_sparse_flow(lk_flow_list, ground_truth_flow)

        # flow_on_image = flow_plot.sparse_flow_as_quiver_plot(lk_flow_list, np.copy(image))
        # tiler.add_image(flow_on_image)

        # flow_on_image = flow_plot.sparse_flow_as_quiver_plot(lk_flow_subtracted, np.copy(image))
        # tiler.add_image(flow_on_image)

        cv2.imshow(session_name, tiler.compose())
        cv2.setWindowTitle(session_name, session_name + ' real fps: {:.1f} sim fps: {:.1f}'.format(
            1.0 / wall_delta_t,
            1.0 / sim_delta_t))
        tiler.clear_scene()
        cv2.waitKey(1)

        last_warped_image = warped_image
        optical_flow_rot_last = optical_flow_rot
        rate_limit.sleep()
        last_sim_t = sim_t
        last_wall_t = wall_t
