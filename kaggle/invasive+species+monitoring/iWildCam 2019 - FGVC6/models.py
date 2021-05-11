
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Flatten, Dense, Conv2D, MaxPool2D, Input, GlobalAveragePooling2D

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import applications

from tensorflow.keras import optimizers


# vgg16 정의
def vgg_func(input_shape, classes_num):
    vgg16 = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # freeze layer
    vgg16.trainable = False

    add_model = Sequential()
    add_model.add(vgg16)
    add_model.add(Flatten())
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dropout(0.2))
    add_model.add(Dense(classes_num, activation='softmax'))

    optimizer = optimizers.SGD(lr=0.01,decay=1e-5, momentum=0.9, nesterov=True)

    add_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    add_model.summary()
    return add_model


#  resnet50 정의
def my_resnet_func(input_shape, num_classes):
    resnet50 = applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    resnet50.trainable = False

    add_model = Sequential()
    add_model.add(resnet50)
    add_model.add(Flatten())
    # add_model.add(Dense(256, activation='relu'))
    add_model.add(Dense(512, activation='relu'))
    add_model.add(Dropout(0.5))
    add_model.add(Dense(num_classes, activation='softmax'))
    # add_model.add(Dense(num_classes, activation='sigmoid'))

    # binary_crossentropy
    add_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return add_model


#  cnn 정의
def my_cnn(input_shape, num_classes):

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile( loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
