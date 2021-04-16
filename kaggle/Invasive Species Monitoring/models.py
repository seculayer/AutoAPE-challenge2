from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications

from tensorflow.keras import optimizers

# cnn 정의
def cnn_func(img_size):
    shape = (img_size[0], img_size[1], img_size[2])
    kernel = (3,3)

    model = Sequential()
    model.add(Conv2D(32, (3,3), kernel, input_shape=shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), kernel, input_shape=shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), kernel, input_shape=shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Flatten())
    # model.add(Activation('relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #  optimizer=RMSprop(lr=1e-4)
    #  optimizer='Adam'
    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    return model


# vgg 정의
def vgg_func():
    base_model = applications.VGG19(include_top=False, weights='imagenet', input_shape=(225, 225, 3))
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer = optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()

    return model


#  resnet 정의
def resnet_func():
    base_model = applications.ResNet152(include_top=False, weights='imagenet', input_shape=(225, 225, 3))

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(1, activation='sigmoid'))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0),
                  metrics=['accuracy'])
    model.summary()

    return model


# inception v3 정의
def inceptionv3_func():
    base_model = applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(225, 225, 3))

    add_model = Sequential()
    add_model.add(BatchNormalization())
    add_model.add(MaxPool2D(pool_size=(2,2)))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(3, activation='sigmoid'))


    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='binary_crossentropy', optimizer = optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()

    return model

