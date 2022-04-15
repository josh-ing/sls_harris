import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def get_outline(image, reference):
    blur = cv.blur(image,(5,5))
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER)
    ret, mask = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    img_contour = np.zeros(image.shape)
    modImage = cv.drawContours(img_contour, contours, -1, (0, 255, 0), 3)

    #reference image
    blur_ref = cv.blur(reference,(5,5))
    gray_ref = cv.cvtColor(blur_ref, cv.COLOR_BGR2GRAY)
    pixel_ref = reference.reshape((-1, 3))
    pixel_ref = np.float32(pixel_ref)
    ret, mask_ref = cv.threshold(gray, 140, 255, cv.THRESH_BINARY)
    contours_ref, hierarchy = cv.findContours(mask_ref, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    ref_contour = np.zeros(reference.shape)
    modReference = cv.drawContours(ref_contour, contours_ref, -1, (0, 255, 0), 3)

    return modImage, modReference
