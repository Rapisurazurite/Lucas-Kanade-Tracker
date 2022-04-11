#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 18:21:26 2022

@author: lazurite
"""


import sys
sys.path.append("build")
import cmake_example
import numpy as np
import cv2
import time
img1 = cv2.imread("res/LK1.png", 0)
img2 = cv2.imread("res/LK2.png", 0)

print(img1.shape)

key_points = cmake_example.detectKeypoint(img1)


time_start = time.time()
success, kp2_mul = cmake_example.OpticalFlowMultiLevel(img1, img2, key_points)
print("time used by multi level: ", time.time() - time_start)
r_img0 = cmake_example.DrawOpticalFlowInImage(img1, key_points, kp2_mul, success)

time_start = time.time()
success, kp2_sig = cmake_example.OpticalFlowSingleLevel(img1, img2, key_points, False, False)
r_img1 = cmake_example.DrawOpticalFlowInImage(img1, key_points, kp2_sig, success)
print("time used by single level: ", time.time() - time_start)

time_start = time.time()
success, pt2 = cmake_example.opticalFlowDetectUsingOpencv(img1, img2, key_points)
print("time used by opencv: ", time.time() - time_start)
pt1 = cmake_example.KeyPointToPoint2f(key_points)
r_img3 = cmake_example.DrawOpticalFlowInImage(img1, pt1, pt2, success)

# cv2.imshow("multiLevel", r_img0)
# cv2.imshow("singleLevel", r_img1)
# cv2.imshow("opencv", r_img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
