import os
import cv2 as cv
from feature_match import align_images

#maybe instead of using orb feature matching, use sobel edge detection or canny
#or use harris corner detection
#feature matches do be kinda not working for complex logos and images
#instead sobel edge detection/ corner detection may be the way to go
# try this https://stackoverflow.com/questions/54589566/how-to-detect-edge-points-of-an-object-using-opencv-python

directory = os.getcwd()
input = os.path.join(directory, 'data\IMG_3782.jpg')
reference = os.path.join(directory, 'data\steelers_reference.jpg')
image = cv.imread(input, cv.IMREAD_COLOR)
template = cv.imread(reference, cv.IMREAD_COLOR)

x, y, z = image.shape
i, j, k = template.shape
if ((x, y) < (i, j)):
    image = cv.resize(image, (i,j))
if ((x, y) > (i, j)):
    template = cv.resize(template, (x, y))

aligned, h, img3 = align_images(image, template)
cv.imwrite("aligned.jpg", aligned)
cv.imwrite("image_matches.jpg", img3)
print("Estimated homography : \n", h)
