import os
import glob

import pandas as pd
import numpy as np
from pathlib import Path

import random
from tqdm.notebook import tqdm
import pydicom # Handle MRI images

import cv2  # OpenCV - https://docs.opencv.org/master/d6/d00/tutorial_py_root.html

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

data_dir = Path('../input/rsna-miccai-brain-tumor-radiogenomic-classification/')

mri_types = ["FLAIR", "T1w", "T2w", "T1wCE"]
excluded_images = [109, 123, 709] # Bad images

train_df = pd.read_csv(data_dir / "train_labels.csv",
#                        index='id',
#                       nrows=100000
                      )
test_df = pd.read_csv(data_dir / "sample_submission.csv")
sample_submission = pd.read_csv(data_dir / "sample_submission.csv")

train_df = train_df[~train_df.BraTS21ID.isin(excluded_images)]

print(f"train data: Rows={train_df.shape[0]}, Columns={train_df.shape[1]}")


def load_dicom(path, size=224):
    '''
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1, then rescales to 0 and 255

    Not super sure if this kind of scaling is appropriate, but everyone seems to do it.
    '''
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    # transform data into black and white scale / grayscale
    #     data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))


def get_all_image_paths(brats21id, image_type, folder='train'):
    '''
    Returns an arry of all the images of a particular type for a particular patient ID
    '''
    assert (image_type in mri_types)

    patient_path = os.path.join(
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder,
        str(brats21id).zfill(5),
    )

    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")),
        key=lambda x: int(x[:-4].split("-")[-1]),
    )

    num_images = len(paths)

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)

    interval = 3

    if num_images < 10:
        interval = 1

    return np.array(paths[start:end:interval])


def get_all_images(brats21id, image_type, folder='train', size=225):
    return [load_dicom(path, size) for path in get_all_image_paths(brats21id, image_type, folder)]


def get_all_data_for_train(image_type, image_size=32):
    global train_df

    X = []
    y = []
    train_ids = []

    for i in tqdm(train_df.index):
        x = train_df.loc[i]
        images = get_all_images(int(x['BraTS21ID']), image_type, 'train', image_size)
        label = x['MGMT_value']

        X += images
        y += [label] * len(images)
        train_ids += [int(x['BraTS21ID'])] * len(images)
        assert (len(X) == len(y))
    return np.array(X), np.array(y), np.array(train_ids)


def get_all_data_for_test(image_type, image_size=32):
    global test_df

    X = []
    test_ids = []

    for i in tqdm(test_df.index):
        x = test_df.loc[i]
        images = get_all_images(int(x['BraTS21ID']), image_type, 'test', image_size)
        X += images
        test_ids += [int(x['BraTS21ID'])] * len(images)

    return np.array(X), np.array(test_ids)


X, y, trainidt = get_all_data_for_train('T1wCE', image_size=32)
X_test, testidt = get_all_data_for_test('T1wCE', image_size=32)

print(X.shape, y.shape, trainidt.shape)


X_train, X_valid, y_train, y_valid, trainidt_train, trainidt_valid = train_test_split(X, y, trainidt, test_size=0.2, random_state=42)

print(X_train.shape)

X_train = tf.expand_dims(X_train, axis=-1)
X_valid = tf.expand_dims(X_valid, axis=-1)

print(X_train.shape)

y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)


# Define, train, and evaluate model
# source: https://keras.io/examples/vision/3D_image_classification/
def get_model01(width=128, height=128, depth=64, name='3dcnn'):
    """Build a 3D convolutional neural network model."""

    inputs = tf.keras.Input((width, height, depth, 1))

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(pool_size=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling3D()(x)
    x = tf.keras.layers.Dense(units=512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = tf.keras.Model(inputs, outputs, name=name)

    # Compile model.
    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    return model


def get_model02():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=X_train.shape[1:])

    h = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    roc_auc = tf.keras.metrics.AUC(name='roc_auc', curve='ROC')

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=[roc_auc]
    )
    return model


def get_model03():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=X_train.shape[1:])

    h = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(64, kernel_size=(4, 4), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(32, kernel_size=(2, 2), activation="relu", name="Conv_2")(h)
    h = keras.layers.MaxPool2D(pool_size=(1, 1))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(32, activation="relu")(h)

    output = keras.layers.Dense(2, activation="softmax")(h)

    model = keras.Model(inpt, output)

    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

    initial_learning_rate = 0.0001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )

    roc_auc = tf.keras.metrics.AUC(name='roc_auc', curve='ROC')

    model.compile(
        loss="categorical_crossentropy",
        #         loss="binary_crossentropy",

        #         optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        optimizer=keras.optimizers.Adam(),

        metrics=[roc_auc],
    )
    return model


checkpoint_filepath = "best_model.h5"

model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor="val_roc_auc",
    mode="max",
    save_best_only=True,
    save_freq="epoch",
    verbose=1,
)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_roc_auc", mode='max', patience=3)

model = get_model03()  # LB score 0.5
model.summary()

history = model.fit(x=X_train, y = y_train, epochs=40,
                    callbacks=[model_checkpoint_cb, early_stopping_cb],
                    validation_data=(X_valid, y_valid))

model_best = tf.keras.models.load_model(filepath=checkpoint_filepath)
y_pred = model_best.predict(X_valid)

pred = np.argmax(y_pred, axis=1)

result = pd.DataFrame(trainidt_valid)
result[1] = pred

result.columns = ["BraTS21ID", "MGMT_value"]
result2 = result.groupby("BraTS21ID", as_index=False).mean()

result2 = result2.merge(train_df, on="BraTS21ID")
auc = roc_auc_score(
    result2.MGMT_value_y,
    result2.MGMT_value_x,
)
print(f"Validation AUC={auc}")

y_pred = model_best.predict(X_test)

pred = np.argmax(y_pred, axis=1) #

result = pd.DataFrame(testidt)
result[1] = pred
print(pred)

result.columns=['BraTS21ID','MGMT_value']

result2 = result.groupby('BraTS21ID',as_index=False).mean()
result2['BraTS21ID'] = sample_submission['BraTS21ID']

# Rounding... 0.907866 -> 0.9
result2['MGMT_value'] = result2['MGMT_value'].apply(lambda x:round(x*10)/10)
# result2['MGMT_value'] = result2['MGMT_value'] # No rounding
result2.to_csv('submission.csv',index=False)

print(result2)