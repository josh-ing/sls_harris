import cv2 as cv
from cv2 import DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.2

def align_images(input, reference):
    inputgray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    referencegray = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)

    # #transformecc helps us calculate rotation of two images and aligns them, might be what we want
    orb = cv.ORB_create(MAX_FEATURES)
    # sift = cv.SIFT_create(MAX_FEATURES)

    # keypoints1, descript1 = sift.detectAndCompute(input, None)
    # keypoints2, descript2 = sift.detectAndCompute(reference, None)
    keypoints1, descript1 = orb.detectAndCompute(inputgray, None)
    keypoints2, descript2 = orb.detectAndCompute(referencegray, None)

    #match features of two images

    bf = cv.DescriptorMatcher_create(DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    # bf = cv.BFMatcher_create(cv.NORM_L1, crossCheck = True)
    matches = bf.match(descript1, descript2, None)
    matches = sorted(matches, key = lambda x: x.distance, reverse=False)

    # good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    # matches = matches[:good_matches]
    matches = matches[:int(len(matches)* GOOD_MATCH_PERCENT)]

    img3 = cv.drawMatches(inputgray, keypoints1, referencegray, keypoints2, matches, None, flags = 2)

    #homography matrix
    pointsA = np.zeros((len(matches), 2), dtype=np.float32)
    pointsB = np.zeros((len(matches), 2), dtype=np.float32)

    for (i, m) in enumerate(matches):
        pointsA[i, :] = keypoints1[m.queryIdx].pt
        pointsB[i, :] = keypoints2[m.trainIdx].pt
    
    (H, mask) = cv.findHomography(pointsA, pointsB, method=cv.RANSAC)
    (height, width) = reference.shape[:2]
    aligned = cv.warpPerspective(input, H, (width, height))

    return aligned, H, img3

def find_centroids(image):
    ret, dest = cv.threshold(image, 0.01 * image.max(), 255, 0)
    dest = np.uint8(dest)

    # find centroids
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(image)
    # define the criteria to stop and refine the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 
                0.001)
    corners = cv.cornerSubPix(image,np.float32(centroids),(5,5), 
              (-1,-1),criteria)
    return corners
    




