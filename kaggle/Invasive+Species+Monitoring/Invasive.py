Load libraries

import pandas as pd
import numpy as np
import os
from os import listdir
from glob import glob
import itertools
import fnmatch
import random
from PIL import Image
import zlib
import itertools
import csv
from tqdm import tqdm
import matplotlib.pylab as plt
import seaborn as sns
import cv2
import skimage
from skimage import transform
from skimage.transform import resize
import scipy
from scipy.misc import imresize, imread
from scipy import misc
import keras
from keras import backend as K
from keras import models, layers, optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, Sequential, model_from_json
from keras.layers import Dense, Dropout, Input, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Lambda, AveragePooling2D
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta, RMSprop
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.utils import class_weight
%matplotlib inline

Read the Files

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import cv2
import math
from glob import glob
import os

master = pd.read_csv("../input/train_labels.csv")
master.head()

img_path = "../input/train/"

y = []
file_paths = []
for i in range(len(master)):
    file_paths.append( img_path + str(master.iloc[i][0]) +'.jpg' )
    y.append(master.iloc[i][1])
y = np.array(y)

file_paths[:10]
y[:10]


Plot the Data

image = cv2.imread(file_paths[0])
plt.figure(figsize=(16,16))
plt.imshow(image)

Read the Data into Arrays

imageSize =256
from tqdm import tqdm
def get_data(file_paths):
    """
    Load the data and labels from the given folder.
    """
    X = []
    for image_filename in tqdm(file_paths):
        img_file = cv2.imread(image_filename)
        if img_file is not None:
            img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
            img_arr = np.asarray(img_file)
            X.append(img_arr)
                           
    X = np.asarray(X)
    return X

X_train = get_data(file_paths)

type(X_train)

X_train = X_train / 255

Split into Train and Validation Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2)
Build the Convolutional Network

pretrained_model_1 = VGG16(include_top=False, input_shape=(imageSize, imageSize, 3))

base_model = pretrained_model_1 # Topless
optimizer1 = keras.optimizers.Adam()
# Add top layer
x = base_model.output
x = Conv2D(256, kernel_size = (3,3), padding = 'valid')(x)
x = Flatten()(x)
x = Dropout(0.75)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
# Train top layer
for layer in base_model.layers:
    layer.trainable = False
model.compile(loss='binary_crossentropy', 
              optimizer=optimizer1, 
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train,y_train, 
                        epochs=10, 
                        batch_size = 32,
                        validation_data=(X_test,y_test), 
                        verbose=1)

del X_train
del y_train
del X_test
del y_test

import gc 
gc.collect()

Prepare the Test Data

sample_submission = pd.read_csv("../input/sample_submission.csv")
img_path = "../input/test/"

test_names = []
file_paths2 = []

for i in range(len(sample_submission)):
    test_names.append(sample_submission.iloc[i][0])
    file_paths2.append( img_path + str(int(sample_submission.iloc[i][0])) +'.jpg' )
    
test_names = np.array(test_names)

file_paths2[:10]

X_test2 = get_data(file_paths2)

X_test2 = X_test2 / 255

y_pred = model.predict(X_test2)

y_pred[:10]

type(y_pred)

y_pred[0]

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()

for i, name in enumerate(test_names):
    sample_submission.loc[sample_submission['name'] == name, 'invasive'] = y_pred[i]

sample_submission.to_csv("submit.csv", index=False)
