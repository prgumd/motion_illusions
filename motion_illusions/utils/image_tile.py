###############################################################################
#
# File: image_tile.py
#
# A session object that constructing an image consiting of several images
# of equal dimension. Allows easy and adaptable visualization of several
# stages of image processing.
#
# History:
# 03-18-20 - Levi Burner - Created file
# 07-28-20 - Levi Burner - Renamed and adjusted for workshop
#
###############################################################################

import numpy as np
import cv2

class ImageTile(object):
    __instances = {}

    @staticmethod
    def get_instance(session=None, max_width=1920, scale_factor=1.0):
        try:
            t = ImageTile.__instances[session]
        except KeyError:
            ImageTile.__instances[session] = ImageTile(max_width, scale_factor)

        return ImageTile.__instances[session]

    def __init__(self, max_width, scale_factor):
        self._images = []

        self._max_res_x = max_width
        self._scale_factor = scale_factor

        self._i_shape = None

    def add_image(self, image):
        width = int(image.shape[1] * self._scale_factor)
        height = int(image.shape[0] * self._scale_factor)
        image = cv2.resize(image, (width, height))

        # Convert from grayscale to color by default, this won't affect any rendering
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        if self._i_shape is None:
            self._i_shape = image.shape
        else:
            if self._i_shape != image.shape:
                raise ValueError('Image of different dimension than first cannot be accepted')

        self._images.append(image)

    def clear_scene(self):
        self._images = []

    def compose(self):
        m = len(self._images)
        i = np.array(self._images)

        max_images_x = int(self._max_res_x / self._i_shape[1])

        rows = [np.hstack(self._images[i:i+max_images_x]) for i in range(0, m, max_images_x)]

        # If the last row is not quite filled
        if rows[-1].shape[1] != rows[0].shape[1]:
            # Three channel image of zeros to fill in till the end of the last row
            zero_pad = np.zeros((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), dtype=np.uint8)
            rows[-1] = np.hstack((rows[-1], zero_pad))

        c = np.vstack(rows)
        return c
