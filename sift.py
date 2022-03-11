import cv2 as cv
from cv2 import MOTION_HOMOGRAPHY
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

img1 = cv.imread('C:\sls_harris\data\DSCF0281.jpg')
imgcolor = cv.imread('C:\sls_harris\data\DSCF0281.jpg')
img2 = cv.imread('C:\sls_harris\data\wisconsin_flat.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, (480, 640))
height, width = img2.shape

#transformecc helps us calculate rotation of two images and aligns them, might be what we want
sift = cv.SIFT_create(MAX_FEATURES)

keypoints1, descript1 = sift.detectAndCompute(img1, None)
keypoints2, descript2 = sift.detectAndCompute(img2, None)

#match features of two images
bf = cv.BFMatcher(cv.NORM_L1, crossCheck = True)
matches = bf.match(descript1, descript2)
matches = sorted(matches, key = lambda x:x.distance, reverse= False)
img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags = 2)
plt.imshow(img3), plt.show()

#now here's the fun part
p1 = np.zeros((len(matches), 2))
p2 = np.zeros((len(matches), 2))

for i in range(len(matches)):
    p1[i, :] = keypoints1[matches[i].queryIdx].pt
    p2[i, :] = keypoints2[matches[i].queryIdx].pt

homography, mask = cv.findHomography(p1, p2, cv.RANSAC)

transformed_img = cv.warpPerspective(imgcolor, homography, (width, height))
plt.imshow(transformed_img)
