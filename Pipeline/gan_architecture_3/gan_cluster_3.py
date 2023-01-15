import os
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
from PIL import Image
from keras.layers import Conv2D, Dropout, Dense, Flatten, Conv2DTranspose, BatchNormalization, LeakyReLU, Reshape
from keras.models import Sequential

### Paths
path = sys.argv[1]  # dataset path 'D:/Medical_Imaging_Zusatz/BreaKHis_v1/Dataset_40X/benign/adenosis/'
imsave_dir = sys.argv[2]  # output path 'D:/Medical_Imaging_Elias/Pipeline/gan_architecture_3/Images_Epoch/'

### Globals
INPUT_SHAPE = 64
NOISE_SHAPE = 100
EPOCHS = sys.argv[3]  # initial 20k
BATCH_SIZE = 16  # 128
PLOT_COUNT = 10

## Load Images
images = []
read = lambda imname: np.asarray(Image.open(imname).convert('L'))  # 'LA', 'L'

for image in os.listdir(path):
    img = read(path + "/" + image)
    # img = cv2.imread(path + folder + "/" + image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (INPUT_SHAPE, INPUT_SHAPE))

    # täusche RGB Image an damit das fürs ResNet als input verwendet werden kann
    img = np.repeat(img[..., np.newaxis], 3, -1)

    images.append(np.array(img))


#Image shape
images = np.array(images) / 255
X_train = images


################################################
################################################


### Generator
generator=Sequential()
generator.add(Dense(4*4*512,input_shape=[NOISE_SHAPE]))
generator.add(Reshape([4,4,512]))
generator.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
generator.add(LeakyReLU(alpha=0.2))
generator.add(BatchNormalization())
generator.add(Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                 activation='sigmoid'))

### Discriminator
discriminator=Sequential()
discriminator.add(Conv2D(32, kernel_size=4, strides=2, padding="same",input_shape=[64,64, 3]))
discriminator.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(BatchNormalization())
discriminator.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1,activation='sigmoid'))


################################################
################################################

GAN = Sequential([generator,discriminator])

discriminator.compile(optimizer='adam',loss='binary_crossentropy')
discriminator.trainable = False

GAN.compile(optimizer='adam',loss='binary_crossentropy')

GAN.layers


################################################
################################################

D_loss=[] #list to collect loss for the discriminator model
G_loss=[] #list to collect loss for generator model

for epoch in range(1, EPOCHS+1):

    if epoch % PLOT_COUNT == 0:

        print(f"Currently on Epoch {epoch}")

    # For every batch in the dataset
    for i in range(X_train.shape[0]//BATCH_SIZE):
        
        # if (i)%100 == 0:

        #     print(f"/tCurrently on batch number {i} of {len(X_train)//BATCH_SIZE}")
            
        noise=np.random.uniform(-1,1,size=[BATCH_SIZE,NOISE_SHAPE])
        
        gen_image = generator.predict_on_batch(noise)
        
        train_dataset = X_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        #train on real image
        train_label=np.ones(shape=(BATCH_SIZE,1))
        discriminator.trainable = True
        d_loss1 = discriminator.train_on_batch(train_dataset,train_label)
        
        #train on fake image
        train_label=np.zeros(shape=(BATCH_SIZE,1))
        d_loss2 = discriminator.train_on_batch(gen_image,train_label)
        
        
        noise=np.random.uniform(-1,1,size=[BATCH_SIZE,NOISE_SHAPE])
        train_label=np.ones(shape=(BATCH_SIZE,1))
        discriminator.trainable = False
        
        #train the generator
        g_loss = GAN.train_on_batch(noise, train_label)
        D_loss.append(d_loss1+d_loss2)
        G_loss.append(g_loss)
        
            
    if epoch % PLOT_COUNT == 0:

        samples = 10
        x_fake = generator.predict(np.random.normal(loc = 0, scale=1, size = (samples,100)))

        for k in range(samples):

            plt.rcParams["figure.figsize"] = (20, 6)
            plt.subplot(2, 5, k+1)
            plt.imshow(x_fake[k].reshape(64,64,3))
            plt.xticks([])
            plt.yticks([])

        plt.savefig(os.path.join(imsave_dir, 'image_at_epoch_{:04d}.png'.format(epoch)))
        # plt.show()
        plt.close()
        
        
    # print('Epoch: %d,  Loss: D_real = %.3f, D_fake = %.3f,  G = %.3f' %   (epoch+1, d_loss1, d_loss2, g_loss))      

print('Training is complete')


Pkl_Filename = f"{imsave_dir}/DCGAN.pkl"

with open(Pkl_Filename, 'wb') as file: 

    pickle.dump(GAN, file)
