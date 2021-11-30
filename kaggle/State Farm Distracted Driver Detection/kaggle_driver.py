import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import glob


train_dir = '../moon/data/driver/imgs/train/'
test_dir = '../moon/data/driver/imgs/test/'


df = pd.read_csv('../moon/data/driver/driver_imgs_list.csv')
sample = pd.read_csv('../moon/data/driver/sample_submission.csv')
print(df.head())
print(sample.head())
print(df.shape[0], len(df['classname'].unique())) # label c0 ~ c9


def gray(dir) :
	images_dir = glob.glob(dir + '*.jpg')
	for i in images_dir :
		image = Image.open(i)
		img = image.convert('L') # L : grayscale
		img.save(i)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1.0/255,
																shear_range = 0.2,
																zoom_range = 0.2,
																horizontal_flip = True,
																validation_split = 0.3)

train_data = train_datagen.flow_from_directory(train_dir, target_size= (224, 224), batch_size= 40, subset='training')
val_data = train_datagen.flow_from_directory(train_dir, target_size= (224, 224), batch_size= 40, subset='validation')
print(train_data.image_shape)
print(val_data.image_shape)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters= 128, kernel_size= (3,3), activation= 'relu', input_shape=(224, 224, 3), padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (3,3)))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3,3), activation= 'relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (3,3)))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (3,3), activation= 'relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size= (3,3)))
model.add(tf.keras.layers.LeakyReLU())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units= 1024, activation= 'relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units= 254, activation= 'relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(units= 10, activation= 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001), metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_data, steps_per_epoch= 15702/40, epochs=10, validation_data= val_data, validation_steps= 6722/40)

def get_data(image_path) :
	img = Image.open(image_path)
	img = img.resize((224, 224), Image.ANTIALIAS)
	return np.array(img)/255

for i, file in enumerate(sample['img']) :
	image = get_data('../moon/data/driver/imgs/test/' + file)
	image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))
	result = model.predict(image)
	sample.iloc[i, 1:] = result[0]

sample.to_csv('submission15.csv', index=False)

