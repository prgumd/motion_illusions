###############################################################################
#
# File: rotation_translation_image_warp.py
#
# Warp an image by a rotation.
# This is a first order approximation, the instantaneous flow due
# to rotation of an observer is used to derive the flow field by
# which the image is warped.
#
# History:
# 07-28-20 - Levi Burner - Created file
#
###############################################################################

from scipy import ndimage
import numpy as np
import cv2

# Calculate an optical flow field due to a body rate rotation
def discrete_optical_flow_due_to_rotation(yaw, pitch, roll, focal_length, grid_size):
    a = pitch
    b = yaw
    g = roll
    f = focal_length

    (x, y) = np.meshgrid(np.arange(0, grid_size[1], dtype=np.float32) - grid_size[1]/2.0 + 0.5,
                         np.arange(0, grid_size[0], dtype=np.float32) - grid_size[0]/2.0 + 0.5)

    # Equations 1 and 2 from page 3 of Passive Navigation as a Pattern Recognition Problem
    u_rot = a*x*y/f - b*(x*x/f + f) + g*y
    v_rot = a*(y*y/f+f) - b*x*y/f - g*x

    # Channels first format
    return np.dstack((u_rot, v_rot))

# Calculate an optical flow field due to a translation in XY plane
# Assumes constant depth of Z = focal_length
def discrete_optical_flow_due_to_2D_translation(u, v, grid_size):
    (uu, vv) = np.meshgrid(np.repeat(v, grid_size[1]),
                           np.repeat(u, grid_size[0]))
    return np.dstack((uu, vv))

def image_warp(image, flow):
    # Need to switch to channels first for handling color images
    # Use atleast_3d to also handle grayscale
    image_rolled = np.moveaxis(np.atleast_3d(image), 2, 0)

    x = np.arange(0, image.shape[1], 1, dtype=np.float32)
    y = np.arange(0, image.shape[0], 1, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    coords = np.array((xv, yv))

    flow_rolled = np.moveaxis(flow, 2, 0)

    # Warp is in the reverse direction so take negative of flow
    mapping = -flow_rolled + coords

    mapping_swapped = np.array((mapping[1, :, :], mapping[0, :, :]))

    image_warped_rolled = np.array([ndimage.map_coordinates(channel, mapping_swapped)
                                       for channel in image_rolled])

    image_warped = np.moveaxis(image_warped_rolled, 0, 2)

    if len(image.shape) == 2:
        image_warped = image_warped.reshape(image.shape)

    return image_warped
