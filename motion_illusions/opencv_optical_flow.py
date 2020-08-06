###############################################################################
#
# File: opencv_optical_flow.py
#
# Wrap OpenCV's optical flow functions to make them even easier to use
#
# History:
# 08-05-20 - Levi Burner - Created file
#
###############################################################################

import cv2
import numpy as np

def lucas_kanade(im1_gray, im2_gray,
                 feature_params = {
                    'maxCorners': 5000,
                    'qualityLevel': 0.1,
                    'minDistance': 7,
                    'blockSize': 7
                 },
                 lucas_kanade_params = {
                    'winSize': (15, 15),
                    'maxLevel': 2,
                    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 }):

    start_points = cv2.goodFeaturesToTrack(im1_gray, mask = None, **feature_params)

    end_points, status, error = cv2.calcOpticalFlowPyrLK(im1_gray, im2_gray, start_points, None, **lucas_kanade_params)

    good_start_points = start_points[status==True]
    good_end_points   = end_points[status==True]

    flow = good_end_points - good_start_points

    return np.concatenate((good_start_points, flow), axis=1)
