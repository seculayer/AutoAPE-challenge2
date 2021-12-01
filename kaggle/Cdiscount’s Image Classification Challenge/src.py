import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import sys, math, io
import bson
import matplotlib.pylot as plt
from skimage.io import imread
import multiprocessing as mp

import struct

%matplotlib inline

import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

data_dir = "../input/cdiscount-image-classification-challenge/"

train_bson_path = os.path.join(data_dir, "train.bson")
num_tran_products = 7069896

test_bson_path = os.path.join(data_dir, "test.bson")
num_test_products = 1768182

categories_path = os.path.join(data_dir, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

categories_df["category_idx"]=pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("categories.csv")
categories_df.head()


def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()

def read_bson(bson_path, num_rescords, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_rescords) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) ==0:
                break
            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data)==length

            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_ims, offset, length]

            if with_categories:
                row+=[item["category_id"]]
            row[product_id]=row

            offset+=length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df

%time train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)

train_offsets_df.to_csv("train_offsets.csv")

train_offsets_df.head()

len(train_offsets_df)
len(train_offsets_df["category_id"].unique())

train_offsets_df["num_imgs"].sum()

def make_val_set(df, split_percentage=0.2, drop_percentage=0.):
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])
    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            keep_size = int(len(product_ids)) * (1. - drop_percentage)
            if keep_size < len(product_ids):
                product_ids = np.random.choice(product_ids, keep_size, replate = False)
            val_size = int(len(product_ids)*split_percentage)
            if val_size > 0 :
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row+[img_idx])
                    else:
                        train_list.append(row+[img_idx])
                pbar.update()
        columns = ["product_id", "category_idx", "img_idx"]
        train_df = pd.DataFrame(train_list, columns = columns)
        val_df = pd.DataFrame(val_list, columns=columns)
        return train_df, val_df

train_images_df, val_images_df = make_val_set(train_offsets_df, split_percentage=0.2, drop_percentage=0.9)

category_idx = 619
num_train = np.sum(train_images_df["category_idx"]==category_idx)
num_val = np.sum(val_images_df["category_idx"]==category_idx)

train_images_df.to_csv("train_images.csv")
val_images_df.to_csv("cal_images.csv")


import imageio

from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

class BSONIterator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class, image_data_generator, lock, target_size=(180,180), with_labels=True, batch_size=32, shuffle=False, seed=None):
        self.file = bson_file
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3.)

        super(BSONIterator, self).__init__(self.samples, batch_size, shuffle, seed)
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array),)+self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
        for i, j in enumerate(index_array):
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])
                item = bson.BSON.decode(item_data)
                img_idx = image_row["img_idx"]
                bson_img = item["imgs"][img_idx]["picture"]

                img = imageio.imread(io.BytesIO(bson_img))

                x=img_to_array(img)
                x=self.image_data_generator.random_transform(x)
                x=self.image_data_generator.standardize(x)

                batch_x[i] = x
                if self.with_labels:
                    batch_y[i, image_row["category_idx"]]=1


            if self.with_labels:
                return batch_x, batch_y
            else:
                return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)

train_bson_path=open(train_bson_path, "rb")

import threading
lock = threading.Lock()

num_classes = 5270
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 128


train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, num_classes, train_datagen, lock, batch_size= batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df, num_classes, val_datagen, lock, batch_size= batch_size, shuffle=True)


next(train_gen)

%time bx, by = next(train_gen)

plt.imshow(bx[-1].astype(np.uint8))



cat_idx = np.argmax(by[-1])
cat_id = idx2cat[cat_idx]
categories_df.loc[cat_id]

%time bx, by = next(val_gen
                    plt.imshow(bx[-1].astype(np.uint8)))

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(180,180,3)))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(128, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

model.fit_generator(train_gen,
                    steps_per_epoch = 10,
                    epochs=3,
                    validation_data = val_gen,
                    validation_stemps = 10,
                    workers = 8)

submission_df = pd.read_csv(data_dir + "sample_submission.csv")
submission_df.head()


num_classes = 5270
num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 128

train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df, num_classes, train_datagen, lock, batch_size= batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df, num_classes, val_datagen, lock, batch_size= batch_size, shuffle=True)

test_datagen = ImageDataGenerator()
data = bson.decode_file_iter(open(test_bson_path, "rb"))

with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        if c >= 1200000:
            product_id = d["_id"]
            num_imgs = len(d["imgs"])

            batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

            for i in range(num_imgs):
                bson_img = d["imgs"][i]["picture"]

                img = imageio.imread(io.BytesIO(bson_img))
                x = img_to_array(img)
                x = test_datagen.random_transform(x)
                x =  test_datagen.standardize(x)

                batch_x[i] = x
        prediction = model.predict(batch_x, batch_size=num_imgs)
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)

        submission_df.iloc[c]["category_id"]= idx2cat[cat_idx]
        pbar.update()

submission_df.to_csv("my_suybmission.csv.gz", compression="gzip", index=False)

