import pandas as pd
import numpy as np
import data_anal, preprocess, features
import os
import multiprocessing
import time
import datetime
from tqdm import tqdm
from pyarrow import csv
import models
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping

from my_models import cid_model, banner_model, app_model, site_model, device_model, date_model

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

FM_EPOCH = 1
DNN_EPOCH = 1
BATCH_SZIE = 16

filename = 'checkpoint-{}-{}-{}-trial-001.h5'.format(FM_EPOCH, DNN_EPOCH, BATCH_SZIE)
early_stopping = EarlyStopping()
checkpoint = ModelCheckpoint(filename,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto'
                             )

callback=[checkpoint, early_stopping]

#  각각의 피쳐 그룹들별로 모델 학습 & predict

# cid = C1,C14,C15,C16,C17,C18,C19,C20,C21
# ./features/feature_C1.csv
# ./test_features/feature_C1.csv
def cid_train():
    data = csv.read_csv('./features/feature_C1.csv').to_pandas()

    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)

    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    feature_dim = x_train.shape[1]
    input_shape = x_train.shape[1:]
    num_classes = 2


    # y_train_fm = np.load('fm_lables.npy')

    print(x_train.shape)
    print(y_train_fm.shape)

    fm_model = cid_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)
    fm_model.save('fm_cid.h5')


    # y_train_dnn = np.load('dnn_lables.npy')
    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = cid_model.my_dnn(input_shape, num_classes, 'fm_cid.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)
    dnn_model.save('dnn_cid.h5')

    test_df =csv.read_csv('./test_features/feature_C1.csv').to_pandas()
    print(len(test_df))
    data = test_df.drop(['id'], axis=1)
    data = data.drop(['index'], axis=1)

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)

    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result


# banner = banner_pos, banner_bag
# ./features/feature_banner_pos.csv
# ./test_features/feature_banner_pos.csv
# ./train_bag/train_banner_bag_features.csv
# ./test_bag/test_banner_bag_features.csv
def banner_pos_train():
    # data = csv.read_csv('./train_bag/train_banner_bag_features.csv').to_pandas()
    data = csv.read_csv('./features/feature_banner_pos.csv').to_pandas()

    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)

    print(data)

    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    feature_dim = x_train.shape[1]
    input_shape = x_train.shape[1:]
    num_classes = 2

    fm_model = banner_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)
    fm_model.save('fm_device.h5')

    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = banner_model.my_dnn(input_shape, num_classes, 'fm_device.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)
    dnn_model.save('dnn_device.h5')


    # test_df =csv.read_csv('./test_bag/test_banner_bag_features.csv').to_pandas()
    test_df =csv.read_csv('./test_features/feature_banner_pos.csv').to_pandas()
    data = test_df.drop(['id'], axis=1)
    data = data.drop(['index'], axis=1)

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)

    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result


# app = app_id,app_domain,app_category
# ./features/feature_app_id.csv
# ./test_features/feature_app_id.csv
def app_train():
    data = csv.read_csv('./features/feature_app_id.csv').to_pandas()

    data = data.drop(['id'], axis=1)
    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)

    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    feature_dim = x_train.shape[1]
    input_shape = x_train.shape[1:]
    num_classes = 2

    fm_model = app_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)
    fm_model.save('fm_app.h5')

    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = app_model.my_dnn(input_shape, num_classes, 'fm_app.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)
    dnn_model.save('dnn_app.h5')

    test_df =csv.read_csv('./test_features/feature_app_id.csv').to_pandas()
    data = test_df.drop(['id'], axis=1)
    data = data.drop(['index'], axis=1)

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)

    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result


# site = site_id,site_domain,site_category
# ./features/feature_site_id.csv
# ./test_features/feature_site_id.csv
def site_train():
    data = csv.read_csv('./features/feature_site_id.csv').to_pandas()

    data = data.drop(['id'], axis=1)
    # data = pd.get_dummies(data, columns=['site_category'])

    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)

    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    print(data)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)


    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    feature_dim = x_train.shape[1]
    input_shape = x_train.shape[1:]
    num_classes = 2

    start = time.time()
    fm_model = site_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)
    fm_model.save('fm_site.h5')
    print("time :", time.time() - start)

    start = time.time()
    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = site_model.my_dnn(input_shape, num_classes, 'fm_site.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)
    dnn_model.save('dnn_site.h5')
    print("time :", time.time() - start)

    test_df =csv.read_csv('./test_features/feature_site_id.csv').to_pandas()
    data = test_df.drop(['id'], axis=1)
    data = data.drop(['index'], axis=1)
    # data = pd.get_dummies(data, columns=['site_category'])

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)

    print(data)

    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result


# device = device_id,device_ip,device_model,device_type,device_conn_type
# 디바이스 타입 2 인애들 행 드롭 + conntype 피쳐는 원핫인코딩
# ./features/feature_device_id.csv
# ./test_features/feature_device_id.csv
def device_train():
    data = csv.read_csv('./features/feature_device_id.csv').to_pandas()

    data = data.drop(['id'], axis=1)
    is_dev_type = data['device_type'] != 2
    data = data[is_dev_type]
    data = pd.get_dummies(data, columns=['device_conn_type'])

    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)


    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    feature_dim = x_train.shape[1]
    input_shape = x_train.shape[1:]
    num_classes = 2

    # y_train_fm = np.load('fm_lables.npy')
    fm_model = device_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)

    fm_model.save('fm_device.h5')

    # y_train_dnn = np.load('dnn_lables.npy')
    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = device_model.my_dnn(input_shape, num_classes, 'fm_device.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)

    dnn_model.save('dnn_device.h5')

    test_df =csv.read_csv('./test_features/feature_device_id.csv').to_pandas()
    data = test_df.drop(['id'], axis=1)
    is_dev_type = data['device_type'] != 2
    data = data[is_dev_type]
    data = pd.get_dummies(data, columns=['device_conn_type'])
    data = data.drop(['index'], axis=1)

    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)


    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result


# ./features/feature_day.csv
# ./test_features/feature_day.csv
# day = date,day  => day 4, date 31 인 행만 학습
def date_train():
    data = csv.read_csv('./features/feature_day.csv').to_pandas()

    data = data.drop(['id'], axis=1)

    is_day = data['day'] == 4
    data = data[is_day]

    data = data.sample(frac=1).reset_index(drop=True)
    data = data.drop(['index'], axis=1)

    # 라벨 생성
    fm_lables = data.loc[:, 'click'].tolist()
    y_train_fm = np.asarray(fm_lables)

    trans_labels = []
    for i in range(len(fm_lables)):
        if fm_lables[i] == 0 :
            trans_labels.append([1, 0])
        else :
            trans_labels.append([0, 1])

    dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
    y_train_dnn = np.asarray(dnn_lables)
    data = data.drop(['click'], axis=1)

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    x_train = np.array(data)
    # input_shape = x_train.shape[0]
    input_shape = x_train.shape[1:]
    feature_dim = x_train.shape[1]
    num_classes = 2


    fm_model = date_model.FM(feature_dim)
    fm_model.fit(x_train, y_train_fm, epochs=FM_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_fm), callbacks=callback)
    fm_model.save('fm_date.h5')

    y_train_dnn = np.reshape(y_train_dnn, (len(y_train_dnn),2))
    dnn_model = date_model.my_dnn(input_shape, num_classes, 'fm_date.h5')
    dnn_model.fit(x_train, y_train_dnn, epochs=DNN_EPOCH, batch_size=BATCH_SZIE, validation_data=(x_train, y_train_dnn), callbacks=callback)
    dnn_model.save('dnn_date.h5')


    test_df =csv.read_csv('./test_features/feature_day.csv').to_pandas()

    data = test_df.drop(['id'], axis=1)
    is_day = data['day'] == 4
    data = data[is_day]
    data = data.drop(['index'], axis=1)


    scaler = MinMaxScaler()
    x_test = scaler.fit_transform(data)

    result = dnn_model.predict(x_test, verbose=1)
    result = result[:,1]

    return result



def history(model):
    # train history 기록
    history_df = pd.DataFrame(model.history)
    history_df[['loss', 'accuracy']].plot()
    history_df[['val_loss', 'val_accuracy']].plot()

    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')
    history_csv_file = str(nowDatetime) + '.csv'

    path = './logs/'

    with open(path+history_csv_file, mode='w') as f:
        history_df.to_csv(f)













