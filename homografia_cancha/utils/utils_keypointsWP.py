import sys
import cv2
import math
import copy
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import linear_sum_assignment
from scipy.stats import linregress
from ellipse import LsqEllipse
from itertools import product
from functools import reduce

from utils.utils_field import _draw_field
from utils.utils_heatmap import generate_gaussian_array_vectorized


class KeypointsWP(object):
    def __init__(self, calibration, size_in=(1920, 1080), size_out=(960, 540)):

        self.keypoint_world_coords_3D = [[-52.5, 34.0, -0.0], [0.0, 34.0, -0.0], [52.5, 34.0, -0.0],
                                         [-52.5, 20.16, -0.0], [-36.0, 20.16, -0.0], [36.0, 20.16, -0.0],
                                         [52.5, 20.16, -0.0], [-52.5, 9.16, -0.0], [-47.0, 9.16, -0.0],
                                         [47.0, 9.16, -0.0], [52.5, 9.16, -0.0], [-52.5, 3.66, 2.44],
                                         [-52.5, 3.66, -0.0], [52.5, 3.66, -0.0], [52.5, 3.66, 2.44],
                                         [-52.5, -3.66, 2.44], [-52.5, -3.66, -0.0], [52.5, -3.66, -0.0],
                                         [52.5, -3.66, 2.44], [-52.5, -9.16, -0.0], [-47.0, -9.16, -0.0],
                                         [47.0, -9.16, -0.0], [52.5, -9.16, -0.0], [-52.5, -20.16, -0.0],
                                         [-36.0, -20.16, -0.0], [36.0, -20.16, -0.0], [52.5, -20.16, -0.0],
                                         [-52.5, -34.0, -0.0], [0.0, -34.0, -0.0], [52.5, -34.0, -0.0],
                                         [-36.0, 7.32, -0.0], [0.0, 9.15, -0.0], [36.0, 7.32, -0.0],
                                         [-36.0, -7.31, -0.0], [0.0, -9.15, -0.0], [36.0, -7.31, -0.0],
                                         [-32.51, 1.71, -0.0], [-8.82, 2.47, -0.0], [8.81, 2.47, -0.0],
                                         [32.5, 1.71, -0.0], [-32.51, -1.7, -0.0], [-8.82, -2.46, -0.0],
                                         [8.81, -2.46, -0.0], [32.5, -1.7, -0.0], [-41.5, -0.0, -0.0],
                                         [-36.0, -0.0, -0.0], [-32.35, -0.0, -0.0], [-6.47, 6.47, -0.0],
                                         [6.47, 6.47, -0.0], [-9.15, -0.0, -0.0], [0.0, -0.0, -0.0],
                                         [9.0, -0.0, -0.0], [-6.47, -6.47, -0.0], [6.47, -6.47, -0.0],
                                         [32.35, -0.0, -0.0], [36.0, -0.0, -0.0], [41.5, -0.0, -0.0]]


        self.calibration = calibration

        self.w_orig, self.h_orig = size_in
        self.w, self.h = size_out
        self.size = (self.w, self.h)
        self.h_extra = self.h * 0.5
        self.w_extra = self.w * 0.5

        self.keypoints_final = {}

        self.num_channels = len(self.keypoint_world_coords_3D) + 1
        self.mask_array = np.ones(self.num_channels).astype(int)


    def get_tensor_w_mask(self):

        self.get_kp_from_calibration()
        heatmap_tensor = generate_gaussian_array_vectorized(self.num_channels, self.keypoints_final, self.size,
                                                            down_ratio=2, sigma=2)
        return heatmap_tensor, self.mask_array


    def get_kp_from_calibration(self):
        K, R, t, dist = self.calibration
        for kp in range(1, len(self.keypoint_world_coords_3D)+1):
            wp = np.array(self.keypoint_world_coords_3D[kp-1])
            img_pt, _ = cv2.projectPoints(wp, R, t, K, dist)
            img_pt = img_pt[0][0]
            img_pt[0] *= self.w / self.w_orig
            img_pt[1] *= self.h / self.h_orig

            in_frame = True if 0 <= img_pt[0] <= self.w and 0 <= img_pt[1] <= self.w else False,

            self.keypoints_final[kp] = {'x': img_pt[0],
                                        'y': img_pt[1],
                                        'in_frame': in_frame,
                                        'close_to_frame': True if -self.w_extra <= img_pt[0] <= self.w + self.w_extra and \
                                                                  -self.h_extra <= img_pt[1] <= self.h + self.h_extra else False}




