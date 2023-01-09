# import necessary packages
import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf

# TODO create a config file for global vars


# global vars
IMG_SHAPE = [256, 256]
NUM_CHANNELS = 1
BATCH_SIZE = 16

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def get_all_images(path):
    files = []
    for folder in os.listdir(path):
        if "." not in folder:
            for image in os.listdir(path + folder):
                img = path + folder + "/" + image
                files.append(img)
    return files

def get_images_single_folder(path):
    files = []
    for img in os.listdir(path):
        files.append(path + img)
    return files


# read images and provide a consistent size
def resize_image(filename):

    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=NUM_CHANNELS)
    image_resized = tf.image.resize(image_decoded, IMG_SHAPE)

    return image_resized


# normalize images
def normalize_images(image):
    image = (image - 127.5) / 127.5

    return image


def prepare_dataset(files: list):
    random.shuffle(files)
    dataset = tf.data.Dataset.from_tensor_slices((files))
    dataset = dataset.shuffle(10000)
    dataset = dataset.map(resize_image)
    dataset = dataset.map(normalize_images, num_parallel_calls=16)
    dataset = dataset.batch(BATCH_SIZE).prefetch(BATCH_SIZE * 4)

    return dataset


# loss function discriminator
def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # [0,0,...,0] with generated images since they are fake
    generated_loss = cross_entropy(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + generated_loss

    return total_loss


# loss function discriminator generator
def generator_loss(generated_output):
    return cross_entropy(tf.ones_like(generated_output), generated_output)


def generate_and_save_images(model, epoch, test_input, imsave_dir):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(8, 8))

#    for i in range(predictions.shape[0]):
 #       plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.title(f'Generated image at epoch: {epoch}')
    plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
   # plt.tight_layout()
    plt.savefig(os.path.join(imsave_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close()


def create_checkpoint(cp_path, gen_opt, disc_opt, gen, disc):
    checkpoint_prefix = os.path.join(cp_path, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=disc_opt,
                                     generator=gen,
                                     discriminator=disc)
    return checkpoint


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


# create discriminator class
class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(256 * 2, (4, 4), strides=(2, 2), padding='same')
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(256 * 4, (4, 4), strides=(2, 2), padding='same')
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(256 * 8, (4, 4), strides=(2, 2), padding='same')
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid')

        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))

        x = self.conv2(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv3(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv4(x)
        x = self.batchnorm4(x, training=training)
        x = tf.nn.leaky_relu(x)

        x = self.conv5(x)
        x = self.fc1(x)

        return x
