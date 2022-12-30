# import necessary packages
import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import PIL
import imageio
import random
# import SimpleITK as sitk
from IPython import display
from IPython.display import Image


# global vars
BENIGN_PATH = '../Dataset/benign/'
MALIGNANT_PATH = '../Dataset/malignant/'


def get_all_images(path):
    files = []
    for folder in os.listdir(path):
        if "." not in folder:
            for image in os.listdir(path + folder):
                img = path + folder + "/" + image
                files.append(img)
    return files


# read images and provide a consistent size
def resize_image(filename, num_channels: int = 1, reshape_size: list = [460, 700]):

    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels = num_channels)
    image_resized = tf.image.resize(image_decoded, reshape_size)

    return image_resized


# normalize images
def normalize_images(image):
    image = (image - 127.5) / 127.5

    return image


def prepare_dataset(files: list):
    random.shuffle(files)
    dataset = tf.data.Dataset.from_tensor_slices((files))
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(resize_image, num_parallel_calls = 16)
    dataset = dataset.map(normalize_images, num_parallel_calls = 16)
    return dataset
