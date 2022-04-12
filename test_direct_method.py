import sys
sys.path.append("build")
import cmake_example
import numpy as np
import cv2
import time

left = cv2.imread("res/left.png", 0)
disparity_img = cv2.imread("res/disparity.png", 0)

pixels_ref, depth_ref = cmake_example.randomSamplePoint(left, disparity_img)

for i in range(1, 6):
    im_name = "res/{:0>6d}.png".format(i)
    img2 = cv2.imread(im_name, 0)
    im, T = cmake_example.directPoseEstimationMultiLayer(left, img2, pixels_ref, depth_ref)

    cv2.imshow("show", im)
    cv2.waitKey(0)

cv2.destroyAllWindows()