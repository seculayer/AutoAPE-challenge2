from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

K = tf.keras.backend



class MyLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim=1, output_dim=30, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.input_dim, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        a = K.pow(K.dot(x,self.kernel), 2)
        b = K.dot(K.pow(x, 2), K.pow(self.kernel, 2))
        return K.mean(a-b, 1, keepdims=True)*0.5

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        return config



def FM(feature_dim):
    inputs = tf.keras.Input((feature_dim,))
    print(inputs)
    liner = tf.keras.layers.Dense(units=1,
                                  bias_regularizer=tf.keras.regularizers.l2(0.01),
                                  kernel_regularizer=tf.keras.regularizers.l1(0.02),
                                  )(inputs)
    cross = MyLayer(feature_dim)(inputs)
    add = tf.keras.layers.Add()([liner, cross])
    predictions = tf.keras.layers.Activation('sigmoid')(add)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['binary_accuracy'])
    return model



def my_dnn(input_shape, num_classes, prev_name):

    prev_model = load_model(prev_name, custom_objects={'MyLayer':MyLayer})
    # prev_model = load_model(prev_name)

    model = Sequential()
    model.add(prev_model)
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(225, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="sigmoid"))
    model.compile( optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
