import tensorflow as tf

from tensorflow.keras.layers import Activation,  Dropout, Flatten, Dense, Conv1D, LSTM
from tensorflow.keras.models import Sequential

#  dnn 정의
def my_dnn(input, num_classes):

    model = Sequential()
    model.add(tf.keras.Input(input))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(225))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('sigmoid'))

    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


#  lstm 정의
def my_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(10, activation = 'relu', input_shape=input_shape))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model