import os
import cv2 as cv
from contours import get_outline
from feature_match import align_images 

#maybe instead of using orb feature matching, use sobel edge detection or canny
#or use harris corner detection
#feature matches do be kinda not working for complex logos and images
#instead sobel edge detection/ corner detection may be the way to go
# try this https://stackoverflow.com/questions/54589566/how-to-detect-edge-points-of-an-object-using-opencv-python

directory = os.getcwd()
input = os.path.join(directory, 'data\IMG_3792.jpg')
reference = os.path.join(directory, 'data\\falcons_reference.jpg')
image = cv.imread(input, cv.IMREAD_COLOR)
template = cv.imread(reference, cv.IMREAD_COLOR)

img_output, ref_output = get_outline(image, template)
aligned, h, img3 = align_images(img_output, ref_output)
cv.imwrite("aligned.jpg", aligned)
cv.imwrite("image_matches.jpg", img3)
print("Estimated homography : \n", h)
# cv.imwrite("mask.jpg", img_output)