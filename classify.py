# Python 3.10.6
import time

# Uses dataset: https://www.kaggle.com/datasets/moltean/fruits?resource=download
# Downloaded 18-Sep-2022

# see requirements.txt for package versions
import tensorflow as tf
import tensorflow.keras.utils as utils

FRUIT_IMAGE_SIZE = (100, 100)
FRUIT_INPUT_SHAPE = (100, 100, 3)

print('[INFO] Loading training data set:')
train = utils.image_dataset_from_directory(
    'fruits-360_dataset/fruits-360/Training',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'rgb',
    batch_size = 32,
    image_size = FRUIT_IMAGE_SIZE,
    shuffle = True, 
    seed = 8008,
)

print('[INFO] Loading validation data set:')
test = utils.image_dataset_from_directory(
    'fruits-360_dataset/fruits-360/Test',
    labels = 'inferred',
    label_mode = 'categorical',
    color_mode = 'rgb',
    batch_size = 32,
    image_size = FRUIT_IMAGE_SIZE,
    shuffle = True,
    seed = 8008,
)

