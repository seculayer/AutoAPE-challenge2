import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from skimage.util import random_noise
from sklearn.model_selection import train_test_split

import cv2
import os
import datetime
import models, preprocess, detect



# Gpu elastic allocation
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

os.chdir("/home/seculayer/yeonjin/kaggle_invasive")


# 트레인 이미지셋 전처리
train_anno = pd.read_csv("./dataset/train_labels.csv")
img_path = "./dataset/resize_train/"

img_list = []
lables = []

for i in range(len(train_anno)):
    img_list.append(img_path + str(train_anno.iloc[i][0]) + ".jpg")
    lables.append(train_anno.iloc[i][1])

noise_path = "./dataset/new_noise_train/"
for i in range(len(train_anno)):
    img_list.append(noise_path + str(train_anno.iloc[i][0]) + ".jpg")
    lables.append(train_anno.iloc[i][1])

lables_arr = np.array(lables) # generate labels


# 노이즈 이미지 생성
i = 0
for idx in range(len(img_list)) :

    #img_file = np.array(Image.open(img_list[i]))
    file_name = img_list[idx]
    img_file = cv2.imread(file_name)
    img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

    i += 1
    noise_img_path = './dataset/new_noise_train/' + str(i) + '.jpg'
    print(f'Processing {noise_img_path} ...')
    if img_file is not None :

        if i % 3 == 0 :
            noise_img = random_noise(img_file, mode='gaussian', var=0.25)
        elif i % 3 == 1 :
            noise_img = random_noise(img_file, mode='s&p', amount=0.25)
        elif i % 3 == 2 :
            noise_img = random_noise(img_file, mode='speckle')

        noise_img = (255*noise_img).astype(np.uint8)
        cv2.imwrite(noise_img_path, img_file)

        plt.title(i)
        plt.imshow(noise_img)
        plt.show()



# 수국 마스킹 & 잎 엣지 뽑아내기
masked_list = []
edge_list = []

i = 0
for file_name in img_list :
    print(i)
    i += 1
    masked_list.append(preprocess.masking(file_name))
np.save('masked.npy', masked_list)


for file_name in tqdm(img_list) :
    edge_list.append(preprocess.tree_edge(file_name))
np.save('edge.npy', edge_list)


# train dataset 리사이징, numpy 변환
train_img = []
for file_name in tqdm(img_list) :
    img_file = cv2.imread(file_name)
    # img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

    if img_file is not None :
        img_file = cv2.resize(img_file, dsize=(225, 225))
        train_img.append(img_file)

train_img = np.asarray(train_img)
train_img = train_img / 255

np.save('train_400.npy', train_img)


# 데이터셋 로드 & 분할
print('dataset load')
train_img = np.load('train_400.npy')
train_img = np.load('masked.npy')

x_train, x_test, y_train, y_test = train_test_split(train_img, lables_arr, test_size=0.2)


# 모델 정의 & 학습
vgg_model = models.vgg_func()

model_json = vgg_model.to_json()
with open("model.json", "w") as json_file :
    json_file.write(model_json)

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("new_noise_model.h5")
loaded_model.compile(loss='binary_crossentropy', optimizer = optimizers.SGD(lr=1e-4, momentum=0.9),
                               metrics=['accuracy'])


epoch = 30
batch_size = 8
img_size = [225, 225, 3]

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
vgg_chkpoint = ModelCheckpoint(filepath='./', monitor='val_loss', verbose=0, save_weights_only=True, mode='auto')

# lr = 1e-4
initial_lr = 0.01
lr_schedule = ExponentialDecay(
    initial_lr,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True
)

data_aug_gen3 = ImageDataGenerator(
    # rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

data_aug_gen3.fit(x_train)
inc_history = vgg_model.fit_generator(
    data_aug_gen3.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=x_train.shape[0] // batch_size,
    epochs=epoch,
    validation_data=(x_test, y_test),
    callbacks=[vgg_chkpoint, early_stop]
)


# predict 실패한 이미지 출력
inc_predict = vgg_model.predict(x_train)
detect.incorrect_lables(x_train, y_train, inc_predict, lables)


# 히스토리 기록
history_df = pd.DataFrame(inc_history.history)

history_df[['loss', 'accuracy']].plot()
history_df[['val_loss', 'val_accuracy']].plot()

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
history_csv_file = 'img400_' + str(nowDatetime) + '.csv'

path = './logs/'

with open(path+history_csv_file, mode='w') as f:
    history_df.to_csv(f)

vgg_model.save_weights("new_noise_model.h5")


# 테스트 데이터셋 전처리
submission = pd.read_csv("./dataset/sample_submission.csv")
test_img_path = "./dataset/test/"

test_names = []
test_img_list = []

for i in range(len(submission)):
    test_names.append(submission.iloc[i][0])
    test_img_list.append(test_img_path + str(int(submission.iloc[i][0])) + '.jpg')

test_names = np.array(test_names)

test_img = []

for file_name in tqdm(test_img_list) :
    img_file = cv2.imread(file_name)
    img_file = cv2.cvtColor(img_file, cv2.COLOR_BGR2RGB)

    if img_file is not None :
        img_file = cv2.resize(img_file, dsize=(225, 225))
        test_img.append(img_file)

np.save('test_400.npy', test_img)

masked_list = []
edge_list = []
i = 0
for file_name in test_img_list :
    print(file_name)
    masked_list.append(preprocess.masking(file_name))
np.save('test_masked.npy', masked_list)



test_img = np.load('test_masked.npy')

test_img = np.asarray(test_img)
test_img = test_img / 255



# prediction
y_pred = loaded_model.predict(test_img)

for i, name in enumerate(test_names):
    submission._set_value(submission['name'] == name,'invasive', y_pred[i] )

submission.to_csv("new_noise_submission.csv", index=False)


