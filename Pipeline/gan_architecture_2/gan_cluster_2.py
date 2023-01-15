### Repo
# - https://www.kaggle.com/code/nageshsingh/generate-realistic-human-face-using-gan

import os
import sys
import time

import cv2
import numpy as np  # linear algebra
from PIL import Image
from keras import Input
from keras.layers import Dense, Reshape, LeakyReLU, Conv2D, Conv2DTranspose, Flatten, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop

# Loading Data
path = sys.argv[1]  # single dataset folder 'D:/Medical_Imaging_Zusatz/BreaKHis_v1/Dataset_40X/benign/adenosis/'
RES_DIR = sys.argv[2]  # output folder '/home/ds21m011/mi/new_gans/gan_architecture_2'
FILE_PATH = '%s/generated_%d.png'
GAN_WEIGHTS_PATH = sys.argv[3]

# Globals
WIDTH = 128
HEIGHT = 128
LATENT_DIM = 32
CHANNELS = 3

ITERATIONS = sys.argv[4]  # initial 15k
BATCH_SIZE = 16
CONTROL_SIZE_SQRT = 6

control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT ** 2, LATENT_DIM)) / 2

if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

## Load Images
images = []
read = lambda imname: np.asarray(Image.open(imname).convert('L')) #'LA', 'L'

for image in os.listdir(path):

    img = read(path + "/" + image)
    # img = cv2.imread(path + folder + "/" + image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (HEIGHT, WIDTH))

    # täusche RGB Image an damit das fürs ResNet als input verwendet werden kann
    img = np.repeat(img[..., np.newaxis], 3, -1)

    images.append(np.array(img))


#Image shape
images = np.array(images) / 255


def create_generator():
    gen_input = Input(shape=(LATENT_DIM, ))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(512, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator



def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        learning_rate=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator



## Create Model
generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM, ))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

optimizer = RMSprop(learning_rate = .0001, clipvalue = 1.0, decay = 1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')




## Training
start = 0
d_losses = []
a_losses = []
images_saved = 0

for step in range(ITERATIONS):

    start_time = time.time()

    latent_vectors = np.random.normal(size = (BATCH_SIZE, LATENT_DIM))
    generated = generator.predict(latent_vectors)

    real = images[start:start + BATCH_SIZE]
    combined_images = np.concatenate([generated, real])

    labels = np.concatenate([np.ones((BATCH_SIZE, 1)), np.zeros((BATCH_SIZE, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(size=(BATCH_SIZE, LATENT_DIM))
    misleading_targets = np.zeros((BATCH_SIZE, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    start += BATCH_SIZE

    if start > images.shape[0] - BATCH_SIZE:

        start = 0

    if step % 50 == 49:

        gan.save_weights(f'{GAN_WEIGHTS_PATH}/gan_{step}.h5')

        print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, ITERATIONS, d_loss, a_loss, time.time() - start_time))

        control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))
        control_generated = generator.predict(control_vectors)
        
        for i in range(CONTROL_SIZE_SQRT ** 2):

            x_off = i % CONTROL_SIZE_SQRT
            y_off = i // CONTROL_SIZE_SQRT
            control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i, :, :, :]

        im = Image.fromarray(np.uint8(control_image * 255))#.save(StringIO(), 'jpeg')
        im.save(FILE_PATH % (RES_DIR, images_saved))
        images_saved += 1
