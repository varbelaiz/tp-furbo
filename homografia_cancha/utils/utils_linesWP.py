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
from utils.utils_heatmap import generate_gaussian_array_vectorized_l, generate_gaussian_array_vectorized_dist_l


class LineKeypointsWP(object):
    def __init__(self, calibration, size_in=(1920, 1080), size_out=(960, 540)):

        self.lines_list = ["Big rect. left bottom",
                           "Big rect. left main",
                           "Big rect. left top",
                           "Big rect. right bottom",
                           "Big rect. right main",
                           "Big rect. right top",
                           "Goal left crossbar",
                           "Goal left post left ",
                           "Goal left post right",
                           "Goal right crossbar",
                           "Goal right post left",
                           "Goal right post right",
                           "Middle line",
                           "Side line bottom",
                           "Side line left",
                           "Side line right",
                           "Side line top",
                           "Small rect. left bottom",
                           "Small rect. left main",
                           "Small rect. left top",
                           "Small rect. right bottom",
                           "Small rect. right main",
                           "Small rect. right top"]

        self.line_extremities = [[[-52.5, -20.16, -0.0], [-36.0, -20.16, -0.0]],
                                 [[-36.0, 20.16, -0.0], [-36.0, -20.16, -0.0]],
                                 [[-36.0, 20.16, -0.0], [-52.5, 20.16, -0.0]],
                                 [[36.0, -20.16, -0.0], [52.5, -20.16, -0.0]],
                                 [[36.0, 20.16, -0.0], [36.0, -20.16, -0.0]],
                                 [[36.0, 20.16, -0.0], [52.5, 20.16, -0.0]],
                                 [[-52.5, -3.66, 2.44], [-52.5, 3.66, 2.44]],
                                 [[-52.5, -3.66, -0.0], [-52.5, -3.66, 2.44]],
                                 [[-52.5, 3.66, -0.0], [-52.5, 3.66, 2.44]],
                                 [[52.5, -3.66, 2.44], [52.5, 3.66, 2.44]],
                                 [[52.5, 3.66, -0.0], [52.5, 3.66, 2.44]],
                                 [[52.5, -3.66, -0.0], [52.5, -3.66, 2.44]],
                                 [[0.0, 34.0, -0.0], [0.0, -34.0, -0.0]],
                                 [[-52.5, -34.0, -0.0], [52.5, -34.0, -0.0]],
                                 [[-52.5, 34.0, -0.0], [-52.5, -34.0, -0.0]],
                                 [[52.5, 34.0, -0.0], [52.5, -34.0, -0.0]],
                                 [[-52.5, 34.0, -0.0], [52.5, 34.0, -0.0]],
                                 [[-52.5, -9.16, -0.0], [-47.0, -9.16, -0.0]],
                                 [[-47.0, -9.16, -0.0], [-47.0, 9.16, -0.0]],
                                 [[-47.0, 9.16, -0.0], [-52.5, 9.16, -0.0]],
                                 [[47.0, -9.16, -0.0], [52.5, -9.16, -0.0]],
                                 [[47.0, -9.16, -0.0], [47.0, 9.16, -0.0]],
                                 [[47.0, 9.16, -0.0], [52.5, 9.16, -0.0]]]

        self.calibration = calibration

        self.w_orig, self.h_orig = size_in
        self.w, self.h = size_out
        self.size = (self.w, self.h)
        self.size_orig = (self.w_orig, self.h_orig)
        self.h_extra = self.h * 0.5
        self.w_extra = self.w * 0.5

        self.lines = {}
        self.lines_aux = {}
        self.num_channels = len(self.lines_list)
        self.mask_array = np.ones(self.num_channels + 1).astype(int)


    def get_tensor_w_mask(self):
        self.get_lines()

        heatmap_tensor = generate_gaussian_array_vectorized_dist_l(self.num_channels,
                                                                   self.lines,
                                                                   self.lines_aux,
                                                                   self.size,
                                                                   self.size_orig,
                                                                   self.calibration,
                                                                   down_ratio=2,
                                                                   sigma=2)


        return heatmap_tensor, self.mask_array


    def get_lines(self):
        for count, line in enumerate(self.line_extremities):
            img_pt1, img_pt2, w_pt1, w_pt2 = self.obtain_extremities(line)

            if img_pt1 is not None and img_pt2 is not None:
                img_pt1[0] *= self.w / self.w_orig
                img_pt1[1] *= self.h / self.h_orig
                img_pt2[0] *= self.w / self.w_orig
                img_pt2[1] *= self.h / self.h_orig
                self.lines[count+1] = {'x_1': img_pt1[0], 'y_1': img_pt1[1],
                                       'x_2': img_pt2[0], 'y_2': img_pt2[1]}

                self.lines_aux[count+1] = {'pt1': w_pt1, 'pt2': w_pt2}


    def obtain_extremities(self, line, num_points=1000):
        w, h = self.w_orig, self.h_orig
        point_1, point_2 = line
        point_1, point_2 = np.array(point_1), np.array(point_2)
        K, R, t, dist = self.calibration

        # Generate linspace for t values (from 0 to 1)
        t_values = np.linspace(0, 1, num_points)

        # Compute the interpolated points between point_1 and point_2
        interpolated_points = np.array([point_1 + t * (point_2 - point_1) for t in t_values])

        ini, fin = None, None
        ini_3D, fin_3D = None, None
        for point in interpolated_points:
            point_2D, _ = cv2.projectPoints(point, R, t, K, dist)
            point_2D = point_2D[0][0]

            if (0 <= point_2D[0] <= w and 0 <= point_2D[1] <= h) and ini is None:
                ini = point_2D
                ini_3D = point
            elif (0 <= point_2D[0] <= w and 0 <= point_2D[1] <= h) and ini is not None:
                fin = point_2D
                fin_3D = point

        return ini, fin, ini_3D, fin_3D





