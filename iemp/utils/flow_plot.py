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
