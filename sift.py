import cv2 as cv
from cv2 import MOTION_HOMOGRAPHY
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15
#maybe instead use homography transformation instead of sift, because sift is for image recognition

directory = os.getcwd()
img1 = cv.imread(os.path.join(directory, 'data\DSCF0276.jpg'))
img2 = cv.imread(os.path.join(directory, 'data\wisconsin_flat.jpg'))
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img2 = cv.resize(img2, (480, 640))

# h, status = cv.findHomography(pts_src, pts_dst)

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

