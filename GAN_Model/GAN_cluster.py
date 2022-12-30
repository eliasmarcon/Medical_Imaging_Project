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

from utils_gan import *

# TODO: add commandline arguments (epoch, batch size, img output dir, model_name)


def main():
    files_benign = get_all_images(BENIGN_PATH)
    print("Sample size of benign images:", len(files_benign))
    files_malignant = get_all_images(MALIGNANT_PATH)
    print("Sample size of malignant images:", len(files_malignant))

    # TODO: add a switch for benign/malignant/all files being generated

    dataset = prepare_dataset(files_benign)




if __name__ == '__main__':
    main()