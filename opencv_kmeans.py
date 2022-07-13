# import libraries and image
import matplotlib.pyplot as plt
import numpy as np
import cv2
path = 'images/london.JPG' # change to 'ofl.png' to segment the ofl screenshot
img = cv2.imread(path)

# preprocess the image
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
twoDimage = img.reshape((-1,3))
twoDimage = np.float32(twoDimage)

# defining some parameters
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5 # 'K' is the number of segments
attempts=10

# apply k-means segmentation
ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((img.shape))

# display the resulting image using pyplot
imgplot = plt.imshow(result_image)
plt.show()