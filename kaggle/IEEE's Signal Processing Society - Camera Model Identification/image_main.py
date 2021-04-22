import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Convolution2D

camera_list = ['HTC-1-M7',
               'LG-Nexus-5x',
               'Motorola-Droid-Maxx',
               'Motorola-Nexus-6',
               'Motorola-X',
               'Samsung-Galaxy-Note3',
               'Samsung-Galaxy-S4',
               'Sony-NEX-7',
               'iPhone-4s',
               'iPhone-6']

X_train, x_val, Y_train, y_val = np.load('./img_data_256_1d_ts33_encode.npy', allow_pickle=True)
# img_data_256_1d_3 -> t:v = 0.33
# img_data_256_1d_ts1 -> t:v = 0.1
# img_data_256_1d_ts2 -> t:v = 0.2
# img_data_256_1d_ts33_encode -> t:v = 0.33 & y label one_hot

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

early_stopping = EarlyStopping(monitor='val_acc', patience = 5)
model_history = model.fit(X_train, Y_train, epochs=30,
                          validation_data=(x_val,y_val), callbacks=[early_stopping])



plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()




X_test = []
sub = pd.read_csv('./img/sample_submission.csv')

for fname in tqdm(sub['fname']):
    filepath = './img/test/' + fname
    X_test.append(img_to_array(load_img(filepath, target_size=(256, 256))))
X_test = np.asarray(X_test)
# X_test = X_test.reshape(-1, 256*256*3)
preds = model.predict(X_test, verbose=1)
preds = np.argmax(preds, axis=1)
preds = [camera_list[p] for p in tqdm(preds)]
print(preds)

sub['camera'] = preds
sub.to_csv('./my_submission32_3_sgd_one_hot.csv', index=False)