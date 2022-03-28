import cv2 as cv
from cv2 import DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def find_corners(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(image)
    dst = cv.cornerHarris(gray, 2, 5, 0.07)
    dst = cv.dilate(dst, None)
    image[dst > 0.01 * dst.max()]=[0, 0, 255]
    return image


    # #list of corners
    # corners_input = find_centroids(dst)


# def find_centroids(image):
#     ret, dest = cv.threshold(image, 0.01 * image.max(), 255, 0)
#     dest = np.uint8(dest)

#     # find centroids
#     ret, labels, stats, centroids = cv.connectedComponentsWithStats(image)
#     # define the criteria to stop and refine the corners
#     criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 
#                 0.001)
#     corners = cv.cornerSubPix(image,np.float32(centroids),(5,5), 
#               (-1,-1),criteria)
#     return corners
