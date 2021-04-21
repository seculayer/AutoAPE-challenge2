from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


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

datagen = ImageDataGenerator(rescale=(1./255))
train_generator = datagen.flow_from_directory(
    './img/train',
    batch_size=1,
    target_size=(256, 256),
    class_mode='sparse')

X_data = []
Y_data = []

## train set의 사진 개수는 총 2750개(폴더 10개, 폴더 당 사진 275개)
for _ in tqdm(range(2750)):
    x, y = train_generator.next()
    X_data.append(x[0])
    Y_data.append(y[0])
X_data = np.asarray(X_data)
Y_data = np.asarray(Y_data, dtype="int32")

plt.imshow((X_data[0]*255).astype('uint8'))
plt.show()

print(X_data.shape, Y_data.shape)
encode_Y = to_categorical(Y_data)

x_train, x_test, y_train, y_test = train_test_split(X_data,encode_Y,
                                                    random_state=42,
                                                    test_size=0.33)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

xy = (x_train, x_test, y_train, y_test)

np.save("./img_data_256_1d_ts33_encode.npy",xy)
