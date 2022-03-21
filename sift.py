import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15

img1 = cv.imread('C:\sls_harris\data\DSCF0275.jpg')
img2 = cv.imread('C:\sls_harris\data\wisconsin_flat.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, (480, 640))

#transformecc helps us calculate rotation of two images and aligns them, might be what we want
sift = cv.SIFT_create()

keypoints1, descript1 = sift.detectAndCompute(img1, None)
keypoints2, descript2 = sift.detectAndCompute(img2, None)

#match features of two images

bf = cv.BFMatcher(cv.NORM_L1, crossCheck = True)
matches = bf.match(descript1, descript2, None)
matches = sorted(matches, key = lambda x: x.distance)

matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches)

for i in range(len(matches)):
    p1 = keypoints1[matches[i].queryIdx].pt
    p2 = keypoints2[matches[i].queryIdx].pt
    print(p1, p2)


#cut off matches
img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches, None, flags = 2)
plt.imshow(img3), plt.show()

