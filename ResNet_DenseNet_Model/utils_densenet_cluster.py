import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import layers
from keras.models import Sequential


# get images function
def get_all_images(path, resize_param):

    images = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

    for folder in os.listdir(path):

        if "." not in folder: 

            for image in os.listdir(path + folder):

                # img = cv2.imread(path + folder + "/" + image)
                img = read(path + folder + "/" + image)
                img = cv2.resize(img, (resize_param, resize_param))

                images.append(np.array(img))

    return images


# build model function
def build_model(backbone, lr = 1e-4):

    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation = 'softmax'))
    
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    
    return model



