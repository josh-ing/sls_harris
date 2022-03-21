import os
import cv2 as cv
from feature_match import align_images


directory = os.getcwd()
input = os.path.join(directory, 'data\IMG_3627_IG.jpg')
reference = os.path.join(directory, 'data\\template_wisconsin.jpg')
image = cv.imread(input, cv.IMREAD_COLOR)
template = cv.imread(reference, cv.IMREAD_COLOR)

aligned, h = align_images(image, template, debug=True)
cv.imwrite("aligned.jpg", aligned)
print("Estimated homography : \n", h)