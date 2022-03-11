import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img1 = cv.imread('C:\sls_harris\data\DSCF0281.jpg')
img2 = cv.imread('C:\sls_harris\data\wisconsin_flat.jpg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

keypoints1, descript1 = sift.detectAndCompute(img1, None)
keypoints2, descript2 = sift.detectAndCompute(img2, None)
#might switch to knn based sift so I can print coordinates 
#perhaps apply ratio test to find coordinates of all keypoints that are matched
bf = cv.BFMatcher_create(cv.NORM_L1, crossCheck = True)
# bf = cv.BFMatcher()
# matches = bf.knnMatch(descript1, descript2, k=2)
matches = bf.match(descript1, descript2)
#ratio test, print this later
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags = 2)
# img3 = cv.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None,**draw_params)
# plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB))
plt.imshow(img3), plt.show()