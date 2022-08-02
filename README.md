# evolv-image-ml
A collection of programs to create and run an image segmentation model using open source code. Developed for Evlov AI.

## Requirements
[segmentation_models](https://github.com/qubvel/segmentation_models) by [qubvel](https://github.com/qubvel)
keras by [tensorflow](https://www.tensorflow.org/install)
[OpenCV](https://opencv.org/releases/)
[Albumentations](https://albumentations.ai/docs/getting_started/installation/)
Numpy:
```zsh
pip install numpy
```
Matplotlib
```zsh
pip install matplotlib
```

## Usage
Training, testing, and validation data must be provided in a data directory with the following files and directories:
data
- test
- testannot
- train
- trainannot
- val 
- valannot
- classes.csv

These directories should contain png files of an image and a corresponding annotated mask.
For example, in train we might have
![abbey_road](abbey_road_example/img.png)
and in trainannot, we would have
![mask](abbey_road_example/label.png)

The index of each image in a directory must coorespond with that of its mask.

In classes.csv, there should be a list of each class and it's corresponding pixel value in the mask, labeled 'class,value'
For example:
class,value
human,[1,1,1]
car,[2,2,2]
tree,[3,3,3]

Once the 'data' directory exists, run the following command to preprocess the images:
```zsh
python3 ./preprocess.py *data-path*
```
where data-path is the path to the data directory.