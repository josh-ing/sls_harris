import cv2 as cv
from cv2 import DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def find_corners(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    dst = cv.dilate(dst, None)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
    # image[dst > 0.01 * dst.max()]=[0, 0, 255]

    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5), 
              (-1,-1),criteria)

    res = np.hstack((centroids,corners))
    res = np.int0(res)
    image[res[:,1],res[:,0]]=[0,0,255]
    image[res[:,3],res[:,2]] = [0,255,0]
    return image


