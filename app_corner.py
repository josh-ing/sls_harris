import os
import cv2 as cv
from harris_corner import find_corners

directory = os.getcwd()
input = os.path.join(directory, 'data\IMG_3631_IG.jpg')
image = cv.imread(input, cv.IMREAD_COLOR)
corners = find_corners(image)
cv.imwrite('harris_corner.jpg', corners)