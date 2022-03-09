import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('wisconsin_flat')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
