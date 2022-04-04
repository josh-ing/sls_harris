import os
import cv2 as cv
from watershed import get_colormask

#maybe instead of using orb feature matching, use sobel edge detection or canny
#or use harris corner detection
#feature matches do be kinda not working for complex logos and images
#instead sobel edge detection/ corner detection may be the way to go
# try this https://stackoverflow.com/questions/54589566/how-to-detect-edge-points-of-an-object-using-opencv-python

directory = os.getcwd()
input = os.path.join(directory, 'data\IMG_3786.jpg')
reference = os.path.join(directory, 'data\\falcons_reference.jpg')
image = cv.imread(input, cv.IMREAD_COLOR)

output = get_colormask(image)
cv.imwrite("mask.jpg", output)