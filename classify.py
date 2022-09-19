# Python 3.10.6
import time

# Uses dataset: https://www.kaggle.com/datasets/moltean/fruits?resource=download
# Downloaded 18-Sep-2022

# see requirements.txt for package versions
import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses

FRUIT_IMAGE_SIZE = (100, 100)
FRUIT_INPUT_SHAPE = (100, 100, 3)

print("\n" * 5)
print(f"Day and time: {time.strftime('%j %H %M %S', time.localtime())}")

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

class Net():
    def __init__(self):
        # Start Sequential Model.  Assuming 100 x 100 x 3 images.
        self.model = models.Sequential()
        # 5 x 5 Conv with stride of 1 and depth of 32 will 
        # take input of 100 x 100 x 3 to 96 x 96 x 32
        self.model.add(layers.Conv2D(32, 5, input_shape = FRUIT_INPUT_SHAPE, activation = 'relu')) 
        # 2 x 2 maxpool will take 96 x 96 x 32 to 48 x 48 x 32
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # 3 x 3 Conv with stride of 1 and depth of 32 will 
        # take input of 48 x 48 x 32 to 46 x 46 x 32
        self.model.add(layers.Conv2D(32, 3, activation = 'relu'))
        # 2 x 2 maxpool will take 46 x 46 x 32 to 23 x 23 x 32
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # 3 x 3 Conv with stride of 1 and depth of 32 
        # take input of 23 x 23 x 32 to 21 x 21 x 32
        self.model.add(layers.Conv2D(32, 3, activation = 'relu'))
        # Add a single row of zero padding on left and bottom to go to 22 x 22 x 32
        self.model.add(layers.ZeroPadding2D(padding = ((0, 1), (0, 1))))
        # 2 x 2 maxpool will take 22 x 22 x 32 to 11 x 11 x 32 
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Time to flatten to 3872
        self.model.add(layers.Flatten())
        # Dense layers, decrease by a factor of 4ish
        self.model.add(layers.Dense(1024, activation = 'relu'))
        self.model.add(layers.Dense(256, activation = 'relu'))
        # There are 131 classes
        self.model.add(layers.Dense(131, activation = 'softmax'))
        self.optimizer = optimizers.SGD(learning_rate = 0.001, momentum = 0.9)
        self.loss = losses.MeanSquaredError()
        self.model.compile(loss = self.loss, optimizer = self.optimizer, metrics = ['accuracy'])
    
    def __str__(self):
        self.model.summary()
        return ""
        
net = Net()
print(net)

results = net.model.fit(
    x = train, 
    validation_data = test,
    shuffle = True, 
    epochs = 100, 
    batch_size = 32, 
    validation_batch_size = 32, 
    verbose = 1
)

print(f"Day and time: {time.strftime('%j %H %M %S', time.localtime())}")
