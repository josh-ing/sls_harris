import cv2 as cv
from cv2 import DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.2
#maybe instead use homography transformation instead of sift, because sift is for image recognition

def align_images(input, reference, debug=False):
    input = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    reference = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
    reference= cv.resize(reference, (640, 480))
    # h, status = cv.findHomography(pts_src, pts_dst)

    #transformecc helps us calculate rotation of two images and aligns them, might be what we want
    orb = cv.ORB_create(MAX_FEATURES)

    keypoints1, descript1 = orb.detectAndCompute(input, None)
    keypoints2, descript2 = orb.detectAndCompute(reference, None)

    #match features of two images

    bf = cv.DescriptorMatcher_create(DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = bf.match(descript1, descript2, None)
    matches = sorted(matches, key = lambda x: x.distance)

    good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:good_matches]

    img3 = cv.drawMatches(input, keypoints1, reference, keypoints2, matches, None, flags = 2)
    plt.imshow(img3), plt.show()

    #homography matrix
    pointsA = np.zeros((len(matches), 2), dtype=np.float32)
    pointsB = np.zeros((len(matches), 2), dtype=np.float32)

    for (i, m) in enumerate(matches):
        pointsA[i, :] = keypoints1[m.queryIdx].pt
        pointsB[i, :] = keypoints2[m.trainIdx].pt

    (H, mask) = cv.findHomography(pointsA, pointsB, cv.RANSAC)
    height, width = reference.shape
    aligned = cv.warpPerspective(input, H, (width, height))

    return aligned, H

    




