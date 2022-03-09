import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

filename = "wisconsin_flat"
img = cv.imread(filename)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
