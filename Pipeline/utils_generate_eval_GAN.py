import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

from utils_generate_eval_GAN import *


# Global Variables
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 2

# # benign types
# ADENOSIS = 'adenosis'
# FIBROADENOMA = 'fibroadenoma'
# PHYLLODES_TUMOR = 'phyllodes_tumor'
# TUBULAR_ADENOMA = 'tubular_adenoma'

# # malignant types
# DUCTAL_CARCINOMA = 'ductal_carcinoma'
# LOBULAR_CARCINOMA = 'lobular_carcinoma'
# MUCINOUS_CARINOMA = 'mucinous_carcinoma'
# PAPILLARY_CARCINOMA = 'papillary_carcinoma'

# get different cancer types
def get_types_array(string):

    if string == "benign":

        # return [ADENOSIS, FIBROADENOMA, PHYLLODES_TUMOR, TUBULAR_ADENOMA]
        return ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']

    elif string == "malignant":

        # return [DUCTAL_CARCINOMA, LOBULAR_CARCINOMA, MUCINOUS_CARINOMA, PAPILLARY_CARCINOMA]
        return ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']


# get different checkpoint paths for cancer types
def get_paths(path_origin, types_array):

    path_array = []
    
    [path_array.append(path_origin + type + "/") for type in types_array]

    return path_array


# generate sample images
def generate_sample_images(generator, generator_optimizer, checkpoint_paths, save_dirs, cancer_types):

    for index, checkpoint_path in enumerate(checkpoint_paths):

        checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, generator = generator)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

        # random vector for image generation 
        random_vector_for_generation = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

        # print(random_vector_for_generation)

        predictions = generator(random_vector_for_generation, training = False)

        for number, prediction in enumerate(predictions):

            fig = plt.figure(figsize=(8, 8))
            plt.imshow(prediction[ :, :, 0] * 127.5 + 127.5, cmap = 'gray')
            plt.axis('off')
            plt.savefig(os.path.join(save_dirs[index], 'GAN_Image_{}_{:04d}.png'.format(cancer_types[index], number)))

            # plt.close()

            # image = np.array(prediction[ :, :, 0] * 127.5 + 127.5)
            # cv2.imwrite(os.path.join(save_dirs[index], 'GAN_Image_{}_{:04d}.png'.format(cancer_types[index], number)), image)



# create generator class
class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(8 * 8 * 256, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2DTranspose(256 * 8, (4, 4), strides=(1, 1), padding='same', use_bias=False)
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(256 * 4, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(256 * 2, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm4 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2DTranspose(256 * 1, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm5 = tf.keras.layers.BatchNormalization()

        self.conv5 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm6 = tf.keras.layers.BatchNormalization()

        self.conv6 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False)

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 8, 8, 256))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.batchnorm4(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv4(x)
        x = self.batchnorm5(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv5(x)
        x = self.batchnorm6(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv6(x))

        return x









