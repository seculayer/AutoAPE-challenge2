import pandas as pd
import numpy as np
import os
import gc
import re
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime
from collections import Counter

MODEL_DIR = './model'
if not os.path.exists(MODEL_DIR): os.mkdir(MODEL_DIR)
filename = os.path.join(MODEL_DIR, 'tmp_checkpoint.h5')

test_df = pd.read_csv('../input/text-normalization-challenge-english-language/en_test_2.csv.zip')
train_df = pd.read_csv('../input/text-normalization-challenge-english-language/en_train.csv.zip')
train_array = train_df.values
test_array = test_df.values
gc.collect()

def make_ord(data_set):
    data_list = []
    for x in data_set:
        x_array = np.ones(27, dtype=int) * 0
        for n, i in zip(list(str(x)), np.arange(27)): x_array[i] = ord(n)              # 영단어에서 영문자 하나씩 가져옴(긴 단어가 있으므로 공간은 27개 줌)
        data_list.append(x_array)                                                       # 가져온 영문자 하나 아스키코드로 변환하고 저장
    data_x = np.array(data_list)
    return data_x

# prepare train_x
train_set = train_array[:9000000,3]
train_x = make_ord(train_set).astype('float32')

# prepare train_y
train_y = train_array[:9000000,2]
print(np.unique(train_y))
l_encoder = LabelEncoder()
train_y = l_encoder.fit_transform(train_y).reshape(-1,1)
oh_encoder = OneHotEncoder()
train_y_oh = oh_encoder.fit_transform(train_y)
train_y_oh = train_y_oh.toarray()
print(train_y.shape)

# prepare pred_target
before_array = test_array[:,2]
pred_target = make_ord(before_array)
pred_target = np.expand_dims(pred_target, -1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(128, 3, input_shape=(27,1)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

model.add(tf.keras.layers.Conv1D(256, 3))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

model.add(tf.keras.layers.Conv1D(512, 3))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.GlobalMaxPooling1D())              

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Dropout(0.7))
model.add(tf.keras.layers.Dense(16, activation='softmax'))
model.summary()

from sklearn.model_selection import StratifiedKFold
s_fold = StratifiedKFold(n_splits=5)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = filename, monitor = 'val_accuracy', verbose = 1, save_best_only = True)

i = 1
for train_index, test_index in s_fold.split(train_x, train_y):
    x_train, y_train = train_x[train_index], train_y[train_index]
    x_valid, y_valid = train_x[test_index], train_y[test_index]

    x_train = np.expand_dims(x_train, -1)
    x_valid = np.expand_dims(x_valid, -1)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=30, batch_size=200, validation_data=(x_valid, y_valid), callbacks=[early_stop, checkpointer])

    
model.load_weights(filename)

pred = model.predict(pred_target)
class_label = oh_encoder.inverse_transform(pred)
print('final:',class_label)

before_array = before_array.reshape(-1,1)

test_set = np.concatenate((class_label, before_array), axis=1)
submission = pd.DataFrame(test_set, columns=['class','before'])
#submission.to_csv("make_class.csv", index=False)
submission.to_csv("make_class_cnn_kfold_7.csv", index=False)