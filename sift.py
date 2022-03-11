import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img1 = cv.imread('C:\sls_harris\data\DSCF0813.jpg')
img2 = cv.imread('C:\sls_harris\data\wisconsin_flat.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints1, descript1 = sift.detectAndCompute(img1, None)
keypoints2, descript2 = sift.detectAndCompute(img2, None)
bf = cv.BFMatcher_create(cv.NORM_L1, crossCheck = True)
matches = bf.match(descript1, descript2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags = 2)
plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.show()