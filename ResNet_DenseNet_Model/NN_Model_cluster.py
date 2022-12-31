import numpy as np
import utils_nn

import gc
from keras import backend as K


BENIGN_PATH = '../dat/benign/'
MALIGNANT_PATH = '../dat/malignant/'
SAVE_MODEL_PATH = './NN_Model/weights.best.hdf5'

benign_images = np.array(utils_nn.get_all_images(BENIGN_PATH, 256))
malignant_images = np.array(utils_nn.get_all_images(MALIGNANT_PATH, 256))

x_train, x_val, y_train, y_val = utils_nn.create_split(benign_images, malignant_images)

train_generator = utils_nn.create_train_generator()

K.clear_session()
gc.collect()

model_densenet = utils_nn.get_model("DenseNet", 256, 3)

# ResNet Modelload
# model_resnet = utils_nn.get_model("ResNet", 256, 3)
    
model = utils_nn.build_model(model_densenet , lr = 1e-4)

# model summary
# model.summary()

# Learning Rate Reducer
learn_control = utils_nn.learning_rate_reducer()
checkpoint = utils_nn.model_checkpoint(SAVE_MODEL_PATH)

history = utils_nn.train_model(model, train_generator, x_train, y_train, x_val, y_val, learn_control, checkpoint)