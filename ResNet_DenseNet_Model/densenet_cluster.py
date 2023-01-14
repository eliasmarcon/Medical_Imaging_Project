# import json
# import math
# import os
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import scipy
# import tensorflow as tf
# import gc
# import itertools

# from PIL import Image
# from functools import partial
# from sklearn import metrics
# from collections import Counter
# from keras import backend as K
# from tqdm import tqdm
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import cohen_kappa_score, accuracy_score

# from keras import layers
# from keras.applications import ResNet50,MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
# from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.optimizers import Adam


import json
import numpy as np
import gc
import random


from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.applications import DenseNet201
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

import utils_densenet_cluster


# Global Variables
BATCH_SIZE = 16
INPUT_SHAPE = 256
EPOCHS = 3


benign_path = 'D:/Medical_Imaging_Zusatz/BreaKHis_v1/Dataset_200X/benign/'
malignant_path = 'D:/Medical_Imaging_Zusatz/BreaKHis_v1/Dataset_200X/malignant/'
# Checkpoint
filepath = "./weights_densenet.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1, save_best_only = True, mode = 'max')


#################################################
#################################################


# Get images
benign_images = np.array(utils_densenet_cluster.get_all_images(benign_path, INPUT_SHAPE))
malignant_images = np.array(utils_densenet_cluster.get_all_images(malignant_path, INPUT_SHAPE))

random.Random(4).shuffle(benign_images)
random.Random(4).shuffle(malignant_images)

# create first split
split = 0.8

benign_train = benign_images[ : int(len(benign_images) * split)]
benign_test = benign_images[int(len(benign_images) * split) : ]
malignant_train = malignant_images[ : int(len(malignant_images) * split)]
malignant_test = malignant_images[int(len(malignant_images) * split) : ]


#################################################
#################################################


### Create Labels
# Create labels
benign_train_label = np.zeros(len(benign_train))
malignant_train_label = np.ones(len(malignant_train))
benign_test_label = np.zeros(len(benign_test))
malignant_test_label = np.ones(len(malignant_test))

# Merge data 
X_train = np.concatenate((benign_train, malignant_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malignant_train_label), axis = 0)
X_test = np.concatenate((benign_test, malignant_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malignant_test_label), axis = 0)

# Shuffle train data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

# Shuffle test data
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

# To categorical
Y_train = to_categorical(Y_train, num_classes = 2)
Y_test = to_categorical(Y_test, num_classes = 2)


#################################################
#################################################


## Train und Eval Split
x_train, x_val, y_train, y_val = train_test_split(
                                                    X_train, Y_train, 
                                                    test_size = 0.2, 
                                                    random_state = 11
                                                )


# Using original generator / data generator
train_generator = ImageDataGenerator(
                                        zoom_range = 2,
                                        rotation_range = 90,
                                        horizontal_flip = True,
                                        vertical_flip = True, 
                                    )


#################################################
#################################################


K.clear_session()
gc.collect()

resnet = DenseNet201(
                        weights = 'imagenet',
                        include_top = False,
                        input_shape = (INPUT_SHAPE, INPUT_SHAPE, 3)
                    )

model = utils_densenet_cluster.build_model(resnet , lr = 1e-4)


#################################################
#################################################


# Learning Rate Reducer
learn_control = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 5,
                                  verbose = 1, factor = 0.2, min_lr = 1e-7)

# Model fit
history = model.fit(
                        train_generator.flow(x_train, y_train, batch_size = BATCH_SIZE),
                        steps_per_epoch = x_train.shape[0] // BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_data = (x_val, y_val),
                        callbacks = [learn_control, checkpoint]
                    )


#################################################
#################################################


# Dump History
with open('./history_40X_64.json', 'w') as f:
    
    json.dump(str(history.history), f)


