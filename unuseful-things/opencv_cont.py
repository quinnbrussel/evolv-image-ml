# import libraries and image
from sys import argv
import cv2 
import matplotlib.pyplot as plt
import numpy as np
path = argv[1]
img = cv2.imread(path)
img = cv2.resize(img,(256, 256))

# preprocess image
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # convert to Grayscale
# compute the threshold of the grayscale image
_, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
# apply edge detection and dilate edges detected
edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

# use the findContours function to find all open and closed regions
# in the image, and store in cnt
# the sorted function orders them so we access the largest first
cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
# mask is a zero pixel mask that has the same shape as the original image
mask = np.zeros((256, 256), np.uint8)
# draw the detected contours on the mask
masked = cv2.drawContours(mask, [cnt], -1, 255, -1)

# segment the regions
dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

# display the results
imgplot = plt.imshow(segmented)
plt.show()