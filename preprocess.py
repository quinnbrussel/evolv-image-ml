from cv2 import imread, imwrite
import os 
import numpy as np


DATA_DIR = 'data'
CLASSES = {'_background_': [0, 0, 0], 'cta': [0, 0, 128]}
values = [value for value in CLASSES.values()]

train_dir = os.path.join(DATA_DIR, 'trainannot')
valid_dir = os.path.join(DATA_DIR, 'valannot')
test_dir = os.path.join(DATA_DIR, 'testannot')

# helper function for image processing
def find_val(pixel):
    [r1, g1, b1] = pixel 
    index = 0
    for value in values:
        [r2, g2, b2] = value 
        if r1 == r2 and g1 == g2 and b1 == b2:
            return [index, index, index]
        else: 
            index += 1
    raise ValueError

# helper function for image processing
def process_images(path, destination):
    for image in os.listdir(path):
        img = imread(path + '/' + image)
        new_image = np.array([[find_val(pixel) for pixel in row] for row in img])
        counter = 0
        imwrite(destination + '/' + str(counter) + '.png', new_image)
        counter += 1

# create processed mask directory
if os.path.exists('processed-masks'):
    files = os.listdir('processed-masks')
    for file in files:
        os.system('rm ' + file)
else:
    os.system('mkdir processed-masks')

os.system('mkdir processed-masks/trainmasks')
os.system('mkdir processed-masks/testmasks') 
os.system('mkdir processed-masks/valmasks')

# Process masks
print('Processing Images', end='')
process_images(train_dir, 'processed-masks/trainmasks')
print('.', end = '')
process_images(valid_dir, 'processed-masks/valmasks')
print('.', end = '')
process_images(test_dir, 'processed-masks/testmasks')
print('.', end = '')