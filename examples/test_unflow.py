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

import cv2
import flowiz as fz
import matplotlib.pyplot as plt
import numpy as np
import torch

from motion_illusions.pytorch_unflow.run import estimate

first_image = '../motion_illusions/pytorch_unflow/images/first.png'
second_image = '../motion_illusions/pytorch_unflow/images/second.png'

if __name__ == '__main__':
    first_tensor =  torch.from_numpy(np.moveaxis(cv2.imread(first_image).astype(np.float32), 2, 0) * 1.0/255.0)
    second_tensor = torch.from_numpy(np.moveaxis(cv2.imread(second_image).astype(np.float32), 2, 0) * 1.0/255.0)

    print('Running Unflow')
    flow = estimate(first_tensor, second_tensor)

    img = fz.convert_from_flow(np.moveaxis(flow.numpy(), 0, 2))
    plt.imshow(img)
    plt.show()
