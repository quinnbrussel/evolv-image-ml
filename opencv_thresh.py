# import libraries and image
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu # note we use scikit-image
import cv2
path = argv[1]
img = cv2.imread(path)

# we convert the image to grayscale
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)

# filter_image takes an image and a grayscale mask
# it multiplies the rgb channel of the image with the grayscale value 
# of the mask, and concatenates the values to form a normal image
def filter_image(image, mask):
    r = image[:,:,0] * mask 
    g = image[:,:,1] * mask 
    b = image[:,:,2] * mask 
    return np.dstack([r, g, b])

# using grayscale version of our image, img_gray, we calculate the 
# threshold used, and use it as a filter
thresh = threshold_otsu(img_gray)
# this creates a mask-like image, img_otsu, that is used to segment 
# the original image
img_otsu = img_gray < thresh 
# now, using img_otsu as our mask, we use our filter_image function
# to create the segmented image
filtered = filter_image(img, img_otsu)

# display results
imgplot = plt.imshow(filtered)
plt.show()