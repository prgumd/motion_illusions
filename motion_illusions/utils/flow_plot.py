###############################################################################
#
# File: flow_plot.py
#
# Plot dense flow
# TODO: plot sparse flow
#
# History:
# 07-28-20 - Levi Burner - Created file
#
###############################################################################

import cv2
import numpy as np

# Generate an HSV image representing the color associated with the direction
# of a vector in a dense flow field
def flow_direction_image(shape=(60,60)):
    (x, y) = np.meshgrid(np.arange(0, shape[1]) - shape[1]/2,
                         np.arange(0, shape[0]) - shape[0]/2)
    theta = np.mod(np.arctan2(x, y) + 2*np.pi, 2*np.pi)
    radius = np.linalg.norm((x, y), axis=0)
    max_radius = np.max(radius)

    flow_hsv = np.zeros((shape[0], shape[1], 3)).astype(np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * radius / max_radius
    flow_hsv[:, :, 2] = 255
    return flow_hsv

# Generate an HSV image using color to represent the gradient direction
# in a optical flow field.
# Assumes channels last input
def visualize_optical_flow(flow):
    theta = np.mod(np.arctan2(flow[:, :, 0], flow[:, :, 1]) + 2*np.pi, 2*np.pi)

    flow_norms = np.linalg.norm(flow, axis=2)
    flow_norms_normalized = flow_norms / np.max(flow_norms)

    flow_hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_hsv[:, :, 0] = 179 * theta / (2*np.pi)
    flow_hsv[:, :, 1] = 255 * flow_norms_normalized
    flow_hsv[:, :, 2] = 255 * (flow_norms_normalized > 0)

    return flow_hsv

# Generate an HSV image using color to represent the gradient direction
# in a optical flow field.
# Assumes channels last input
def visualize_optical_flow_rgb(flow):
    return cv2.cvtColor(visualize_optical_flow(flow), cv2.COLOR_HSV2RGB)

def subtract_dense_flow_from_sparse_flow(flow_sparse, flow_dense):
    flow_dense_sparse = flow_dense[flow_sparse[:, 1].astype(np.int64),
                                   flow_sparse[:, 0].astype(np.int64)]

    flow_sparse_subtracted = flow_sparse[:, 2:4] - flow_dense_sparse
    return np.concatenate((flow_sparse[:, 0:2], flow_sparse_subtracted), axis=1)

def downsample_dense_flow(flow, scale_factor):
    flow_scaled = cv2.resize(flow, None, fx=scale_factor[1], fy=scale_factor[0])
    return flow_scaled

def dense_flow_to_sparse_flow_list(flow, coordinate_scale=1.0):
    x = np.arange(0, flow.shape[1], 1, dtype=np.float32) * coordinate_scale + coordinate_scale / 2.0
    y = np.arange(0, flow.shape[0], 1, dtype=np.float32) * coordinate_scale + coordinate_scale / 2.0
    xv, yv = np.meshgrid(x, y)

    xv = xv.flatten()
    yv = yv.flatten()
    flow_x = flow[:, :, 0].flatten()
    flow_y = flow[:, :, 1].flatten()

    flow_list = np.stack((xv, yv, flow_x, flow_y), axis=1)

    return flow_list

# Plot a dense flow field on an image
def dense_flow_as_quiver_plot(flow, image=None, scale_factor=(0.05, 0.05), quiver_scale=1.0, color=(255, 0, 0), thickness=1):
    if image is None:
        image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)

    flow_scaled = downsample_dense_flow(flow, scale_factor)
    flow_list =  dense_flow_to_sparse_flow_list(flow_scaled, coordinate_scale=image.shape[0]/flow_scaled.shape[0])
    return sparse_flow_as_quiver_plot(flow_list, image, quiver_scale, color, thickness)

# Accept a list of optical flow vectors, plot them as arrows overlayed on an image
def sparse_flow_as_quiver_plot(flow_list, image, quiver_scale=1.0, color=(255,0,0), thickness=1):
    for flow in flow_list:
        start = flow[0:2]
        end = flow[0:2] + quiver_scale * flow[2:4]

        cv2.line(image, tuple(start), tuple(end), color, thickness)

        angle = np.arctan2(flow[2], flow[3])
        mag = quiver_scale * np.linalg.norm(flow[2:4])

        angle_left = angle + np.pi / 4
        angle_right = angle - np.pi / 4

        tip_left_head  = end - 0.5 * mag * np.array((np.sin(angle + np.pi/4), np.cos(angle + np.pi/4)))
        tip_right_head = end - 0.5 * mag * np.array((np.sin(angle - np.pi/4), np.cos(angle - np.pi/4)))

        cv2.line(image, tuple(end), tuple(tip_left_head.astype(np.int32)), color, thickness)
        cv2.line(image, tuple(end), tuple(tip_right_head.astype(np.int32)), color, thickness)

    return image
