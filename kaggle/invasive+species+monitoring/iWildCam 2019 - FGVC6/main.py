import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn

import models, pca_detect, preprocess, megadetect


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


# 트레인 데이터셋 이미지 전처리 + 리사이징 => 넘파이 변환
label_df = pd.read_csv('./train.csv')
train_resized_imgs = []

for image_id in tqdm(label_df['id']):
    train_resized_imgs.append(preprocess.img_pre_processing(image_id, 'train_images'))
print('train image preprocessing completed.')

X_train = np.stack(train_resized_imgs)
np.save('x_train_224.npy', X_train)
print('x train numpy file loading....')
x_train = np.load('./x_train_224.npy')


# 트레인 데이터셋 라벨 생성
label_df = pd.read_csv('./train.csv')
real_label = label_df['category_id']
y_train = []

for i in range(x_train.shape[0]):
    print(f'y train lables processing {i}....')
    label_val = real_label[i]
    temp_list = [0 for _ in range(23)]
    temp_list[label_val] = 1
    y_train.append(temp_list)

y_train = np.array(y_train)
np.save('y_train_224.npy', y_train)


# 테스트 데이터셋 변환
submission_df = pd.read_csv('./sample_submission.csv')
test_resized_imgs = []

for image_id in tqdm(submission_df['Id']):
    test_resized_imgs.append(preprocess.test_img_pre_processing(image_id, 'test_images'))
print('test image preprocessing completed.')

X_test = np.stack(test_resized_imgs)
np.save('x_test_224.npy', X_test)


image_size = 224


x_train = np.load('./x_train_224.npy')
x_train = x_train.astype('float32')
x_train /= 255.
y_train = np.load('./y_train_224.npy')


#  모델 정의 & 학습
batch_size = 32
num_classes = 23
epochs = 30
val_split = 0.1


input_shape = x_train.shape[1:]

# model = models.deep_cnn(input_shape, num_classes)
model = models.my_resnet_func(input_shape, num_classes)
# model = models.vgg_func(input_shape, num_classes)

early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=4)
chkpoint = ModelCheckpoint(filepath='./', monitor='val_loss', verbose=0, save_weights_only=True, mode='auto')
callback = [chkpoint, early_stop]

print(f'x train shape: {x_train.shape}, y train shape: {y_train.shape}')

his = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callback,
    # callbacks = [f1_metrics],
    validation_split=val_split,
    shuffle=True
)


# train history 기록
history_df = pd.DataFrame(his.history)
history_df[['loss', 'accuracy']].plot()
history_df[['val_loss', 'val_accuracy']].plot()

now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
history_csv_file = 'img224_' + str(nowDatetime) + '.csv'

path = './logs/'

with open(path+history_csv_file, mode='w') as f:
    history_df.to_csv(f)


save_dir = os.path.join(os.getcwd(), 'models')
model_name = f'img224_cnn.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
model.save(model_path)


# train dataset predict & 틀린 애들 출력
train_prediction = model.predict(x_train)
cm = confusion_matrix(y_train.argmax(axis=1), train_prediction.argmax(axis=1))
df_cm = pd.DataFrame(cm, range(14), range(14))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": 6})
plt.show()

label_df = pd.read_csv('./train.csv')
y_train_lable = y_train.argmax(axis=1)
train_pred_lable = train_prediction.argmax(axis=1)
pca_detect.incorrect_lables(x_train, y_train_lable, train_pred_lable, label_df)


#  test dataset prediction
x_test = np.load('./x_test_224.npy')
x_test = x_test.astype('float32')
x_test /= 255.

input_shape = x_test.shape[1:]
num_classes = 23

model = models.deep_cnn(input_shape, num_classes)
model.load_weights('./models/img224_cnn.h5')

print('Predict test images.....')
y_test = model.predict(x_test)

result_df = pd.read_csv('./sample_submission.csv')
result_df['Predicted'] = y_test.argmax(axis=1)
print(result_df)

result_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
del result_df['Category']

print('Make submission file.....')
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
result_df.to_csv(f'submission_{image_size}_{nowDatetime}.csv', index=False)

print('Complete.')





