# import libraries and image
from sys import argv 
import cv2 
path = argv[1]
img = cv2.imread(path) 
import matplotlib.pyplot as plt


# preprocess the image
# We comverd the image to rgb then to the HSV colorspace
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

# define the color range to be detected
# This depends on the image we want to look at
light_blue = (90, 70, 50)
dark_blue = (128, 255, 255)
light_green = (40, 40, 40)
dark_green = (70, 255, 255)
mask = cv2.inRange(hsv_img, light_blue, dark_blue)
# mask = cv2.inRange(hsv_img, light_green, dark_green)

# apply to the mask
result = cv2.bitwise_and(img, img, mask=mask) 

imgplot = plt.imshow(result)
plt.show()