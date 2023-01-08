import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

from PIL import Image

from keras import layers
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50, MobileNet, DenseNet201, InceptionV3, NASNetLarge, InceptionResNetV2, NASNetMobile
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from sklearn.model_selection import train_test_split



# global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 2

GAN_BENIGN_PATH = 'D:/Medical_Imaging_Zusatz/Gan_Images/benign/'
GAN_MALIGNANT_PATH = 'D:/Medical_Imaging_Zusatz/Gan_Images/malignant/'

GT_BENIGN_PATH = 'D:/Medical_Imaging_Zusatz/Dataset/benign/'
GT_MALIGNATN_PATH = 'D:/Medical_Imaging_Zusatz/Dataset/malignant/'


# get all images
def get_all_images(path_wish):

    if path_wish == 'gan_benign':

        path = GAN_BENIGN_PATH

    elif path_wish == 'gan_malignant':
        
        path = GAN_MALIGNANT_PATH

    elif path_wish == 'gt_benign':

        path = GT_BENIGN_PATH

    elif path_wish == 'gt_malignant':

        path = GT_MALIGNATN_PATH

    images = []
    read = lambda imname: np.asarray(Image.open(imname).convert('L')) #'LA', 'L'

    for folder in os.listdir(path):

        for image in os.listdir(path + folder):

            img = read(path + folder + "/" + image)
            # img = cv2.imread(path + folder + "/" + image)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            # täusche RGB Image an damit das fürs ResNet als input verwendet werden kann
            img = np.repeat(img[..., np.newaxis], 3, -1)

            images.append(np.array(img))

    return np.array(images)


# create train and test split 
def create_split(benign_images, malignant_images, split = 0.8, test_size = 0.2):

    benign_train = benign_images[ : int(len(benign_images) * split)]
    benign_test = benign_images[int(len(benign_images) * split) : ]
    malignant_train = malignant_images[ : int(len(malignant_images) * split)]
    malignant_test = malignant_images[int(len(malignant_images) * split) : ]

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

    x_train, x_val, y_train, y_val = train_test_split(
                                                        X_train, Y_train, 
                                                        test_size = test_size, 
                                                        random_state = 11
                                                    )

    return X_test, Y_test, x_train, x_val, y_train, y_val


# display some images
def display_images(x_train, y_train, title):

    # Display first 15 images of moles, and how they are classified
    fig = plt.figure(figsize = (15, 15))
    
    columns = 4
    rows = 3

    for i in range(1, columns*rows + 1):
        
        ax = fig.add_subplot(rows, columns, i)
        
        if np.argmax(y_train[i]) == 0:
        
            ax.title.set_text('Benign')
        
        else:
        
            ax.title.set_text('Malignant')
        
        plt.imshow(x_train[i], interpolation = 'nearest')
        
    fig.suptitle(title, fontsize = 16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.show()



# build model
def build_model(backbone, lr = 1e-4):

    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation = 'softmax'))
    
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = Adam(learning_rate = lr),
        metrics = ['accuracy']
    )
    
    return model



# Using original generator
def create_train_generator(range = 90):

    train_generator = ImageDataGenerator(
                                            zoom_range = 2,  # set range for random zoom
                                            rotation_range = range,
                                            horizontal_flip = True,  # randomly flip images
                                            vertical_flip = True,  # randomly flip images
                                        )

    return train_generator



# get model
def get_model(modelname, input_shape, channels):

    if modelname == 'DenseNet':

        model = DenseNet201(
                            weights = 'imagenet',
                            include_top = False,
                            input_shape = (input_shape, input_shape, channels)
                        )

    if modelname == 'ResNet':

        model = ResNet50(
                        weights = 'imagenet',
                        include_top = False,
                        input_shape = (input_shape, input_shape, channels)
                    )

    return model


# get learning rate reducer
def learning_rate_reducer(monitor = 'val_accuracy', min_lr = 1e-7):

    # Learning Rate Reducer
    learn_control = ReduceLROnPlateau(monitor = monitor, patience = 5,
                                      verbose = 1, factor = 0.2, min_lr = min_lr)


    return learn_control


# Checkpoint
def model_checkpoint(filepath):
    # Checkpoint
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_accuracy', verbose = 1, 
                                 save_best_only = True, mode = 'max')

    return checkpoint




# train model
def train_model(model, train_generator, x_train, y_train, x_val, y_val, learn_control, checkpoint):

    history = model.fit(
                        train_generator.flow(x_train, y_train, batch_size = BATCH_SIZE),
                        steps_per_epoch = x_train.shape[0] // BATCH_SIZE,
                        epochs = EPOCHS,
                        validation_data = (x_val, y_val),
                        callbacks = [learn_control, checkpoint]
                    )

    return history



# plot metrics
def plot_metrics(history):

    history_df = pd.DataFrame(history.history)
    history_df[['accuracy', 'val_accuracy']].plot()
    history_df[['loss', 'val_loss']].plot()

    plt.show()
    plt.close()












