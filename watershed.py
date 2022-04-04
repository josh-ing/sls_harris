import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_colormask(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv.dilate(opening, kernel, iterations= 3)

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)


    ret, markers = cv.connectedComponents(sure_fg)

    markers = markers+1

    markers[unknown==255] = 0
    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image