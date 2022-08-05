# evolv-image-ml
A collection of programs to create and run an image segmentation model using open source code. Developed for Evlov AI.

## Requirements
- [segmentation_models](https://github.com/qubvel/segmentation_models) by [qubvel](https://github.com/qubvel)  
- [tensorflow](https://www.tensorflow.org/install)  
- [OpenCV](https://opencv.org/releases/)  
- [Albumentations](https://albumentations.ai/docs/getting_started/installation/)  
- Numpy
```zsh
pip install numpy
```
- Matplotlib
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

These directories should contain png files of an image and a corresponding annotated mask.
For example, in train we might have

![abbey_road](abbey_road_example/img.png)

and in trainannot, we would have

![mask](abbey_road_example/label.png)

The index of each image in a directory must coorespond with that of its mask.

For annotation, [Labelme](https://github.com/wkentaro/labelme), a free annotation software, is a recommended software.

Next, the user must set 'DATA_DIR' in line 1 of header.py to the path of the data directory.

For example:

```python
DATA_DIR = 'data'
```

Then, the user must edit the 'classes' dictionary in line 2 of header.py so that each bin of the dictionary is a segmentation class, and each bin contains the pixel value of the class within the mask.

For example:

```python
CLASSES = {'_background_': [0, 0, 0], 'cta': [0, 0, 128]}
```

The user must now edit 'BATCH_SIZE' in line 3 so that the value reflects the number of images processed per batch. Next edit 'PREDICT_CLASS' so that it is a list containing the classes we want the model to identify. Finally, edit 'EPOCHS' so that it reflects the number of EPOCHS to run during training.

For example:
```python
BATCH_SIZE = 8
PREDICT_CLASS = ['cta']
EPOCHS = 40
```

Once the 'data' directory exists, run the following command to preprocess the images:
```zsh
python3 ./preprocess.py
```

To train a model, enter:
```zsh 
python3 ./unet_training.py
```
This will output a directory, final-model, in the current directory and contains the final model.

Finally, to run this model to segment on a new image, create a directory containing the image to evaluate and run:
```zsh
python3 ./run.py *dirname*
```
where dirname is the name of the directory containing the image.

This will display a mask of the image.