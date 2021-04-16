import pandas as pd
import numpy as np
import data_anal, preprocess, features, train
import os
import multiprocessing

from tqdm import tqdm
from pyarrow import csv
import models
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


#  train dataset 총 40428967개 데이터
#  test dataset 총 3230324개 데이터

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.chdir("/home/seculayer/yeonjin/kaggle_adv")


# train set csv 파일 분할
df = pd.read_csv('./dataset/train.csv')
n = 1100000
list_df = [df[i:i+n] for i in range(0, df.shape[0], n)]

for i in range(len(list_df)):
    name = './split_test/split_test_' + str(i) + '.csv'
    print(name)
    list_df[i].to_csv(name, index=False)


# 멀티프로세싱 전처리 /  path list 생성
path_list = []

tmp = []
for i in range(0, 102):
    tmp.append('./split/split_train_' + str(i) + '.csv')
path_list.append(tmp)


tmp = []
for i in range(102, 204):
    tmp.append('./split/split_train_' + str(i) + '.csv')
path_list.append(tmp)

tmp = []
for i in range(204, 306):
    tmp.append('./split/split_train_' + str(i) + '.csv')
path_list.append(tmp)

tmp = []
for i in range(306, 405):
    tmp.append('./split/split_train_' + str(i) + '.csv')
path_list.append(tmp)


print("Multiprocessing....")
pool = multiprocessing.Pool(processes=4)
pool.map(preprocess.multi_calc, path_list)
pool.close()
pool.join()
print("Multiprocessing completed.")


# # 일반 전처리
# print('train dataset preprocessing...')
# id_list = []
# for i in range(405):
#     preprocess.run_preprocess(i)


# 전처리된 애들 모두 concat
print('splited train dataset concating...')
concat_df =  csv.read_csv('./train_preprocess/new_prepros_0.csv').to_pandas()

for i in range(1, 405):
    path = './train_preprocess/new_prepros_' + str(i) + '.csv'
    print(path)
    tmp_df = csv.read_csv(path).to_pandas()
    concat_df = pd.concat([concat_df, tmp_df])

concat_df.to_csv('./new_concated.csv', index=False)



# 테스트 데이터셋 전처리 + concat
print('test dataset preprocessing & concating...')

num = 40
for i in range(num):
    preprocess.run_test_preprocess(i)

path_list = []
tmp = []
for i in range(0, 10):
    tmp.append('./split_test/split_test_' + str(i) + '.csv')
path_list.append(tmp)


tmp = []
for i in range(10, 20):
    tmp.append('./split_test/split_test_' + str(i) + '.csv')
path_list.append(tmp)

tmp = []
for i in range(20, 30):
    tmp.append('./split_test/split_test_' + str(i) + '.csv')
path_list.append(tmp)

tmp = []
for i in range(30, 46):
    tmp.append('./split_test/split_test_' + str(i) + '.csv')
path_list.append(tmp)


print("Test dataset Multiprocessing....")
pool = multiprocessing.Pool(processes=4)
pool.map(preprocess.multi_clac_test, path_list)
pool.close()
pool.join()
print("Test dataset Multiprocessing completed.")



print("preprocessed test dataset concating...")
test_concat_df =  csv.read_csv('./test_preprocess/new_prepros_0.csv').to_pandas()

for i in range(1, 46):
    path = './test_preprocess/new_prepros_' + str(i) + '.csv'
    tmp_df = csv.read_csv(path).to_pandas()
    test_concat_df = pd.concat([test_concat_df, tmp_df])

test_concat_df.to_csv('./new_test_concated.csv', index=False)
test_concat_df = test_concat_df.drop(['level_0'])
print(test_concat_df)

print('preprocessing compeleted.')


# 라벨 분리 & npy 변환 저장
print("Label split...")
train_df = csv.read_csv('./new_concated.csv').to_pandas()

# fm 모델 라벨
fm_lables = train_df.loc[:, 'click'].tolist()
fm_lables = np.asarray(fm_lables)
np.save('fm_lables.npy', fm_lables)

# dnn 모델 라벨
trans_labels = []
for i in range(len(fm_lables)):
    if fm_lables[i] == 0 :
        print(f'{i} / {len(fm_lables)}')
        trans_labels.append([1, 0])
    else :
        print(f'{i} / {len(fm_lables)}')
        trans_labels.append([0, 1])

dnn_lables = np.reshape(trans_labels, (len(trans_labels), 2))
dnn_lables = np.asarray(dnn_lables)
np.save("dnn_lables.npy", dnn_lables)


# 추가 전처리
# # 1. hour 컬럼 드롭 ( device model ? )
# print("Hour column dropping....")
train_df = csv.read_csv('./new_concated.csv').to_pandas()
# train_df= train_df.drop(['hour'], axis=1)
#
test_df = csv.read_csv('./new_test_concated.csv').to_pandas()
# test_df= test_df.drop(['hour'], axis=1)


# 2. 중복 제거
train_df = train_df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
test_df = test_df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)


# # 4-1. app id, site id 드롭..
# test_df= test_df.drop(['app_id'], axis=1)
# test_df= test_df.drop(['site_id'], axis=1)


# 5. 샘플링
# 클릭안함 0 랜덤 다운 샘플링 / 클릭함 1 랜덤 오버 샘플링
not_click = train_df['click'] == 0
not_clk_df = train_df[not_click]
not_clk_df = not_clk_df.sample(frac=0.9, replace=True)

is_click = train_df['click'] == 0
is_clk_df = train_df[is_click]

sampled_df = pd.concat([not_clk_df, is_clk_df])


# 6. 정규화

print("Normalazation...")
np.random.seed(0)
df_minmax_norm = (train_df - train_df.min()) / ( train_df.max() - train_df.min())
train_df.to_csv('./new_train_normal.csv', index=False)

df_minmax_norm = (test_df - test_df.min()) / ( test_df.max() - test_df.min())
test_df.to_csv('./new_test_normal.csv', index=False)


# train dataset & test dataset 피쳐별로 스플릿
features.feature_split()

# feature bag 생성
features.train_feature_bag()


# test 데이터셋 피쳐별로 학습 & predict
results = []
results.append(train.cid_train())
results.append(train.app_train())
results.append(train.site_train())
results.append(train.device_train())
results.append(train.date_train())
results.append(train.banner_pos_train())

result_df = pd.DataFrame(results)
results = (result_df.sum()).tolist()

# 결과 앙상블
length = len(results)
final_result = list(map(lambda x : x/length, results))

print("Submission ...")
result_df = pd.read_csv('./dataset/sampleSubmission.csv')
result_df.loc[:, "click"] = pd.Series(final_result[:])

sample_df = pd.read_csv('./dataset/sampleSubmission.csv')
tmp_df = sample_df[len(result_df)+1:]
result_df = pd.concat([result_df, tmp_df])
result_df = result_df.fillna(0.5)

result_df.to_csv("final_submission.csv", index=False)














# 모델 정의 & 학습
# # lables = np.load('./final_lables.npy')
# lables = trans_labels
#
# x_train = np.array(train_df)
# x_train = x_train.reshape(len(x_train), 1, x_train.shape[1])
# lables = lables.reshape(len(lables), 1, lables.shape[1])
#
# inp_shape = (1, 7)
# num_classes = 2
# EPOCH = 10
# BATCH_SIZE = 16
#
# model = models.my_dnn(inp_shape, num_classes)
#
# filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(EPOCH, BATCH_SIZE)
# early_stopping = EarlyStopping()
# checkpoint = ModelCheckpoint(filename,
#                              monitor='loss',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='auto'
#                              )
#
# model.fit(x_train, lables, epochs=EPOCH, verbose=1, callbacks=[checkpoint, early_stopping])
# model.save("dnn_model.h5")



# 프리딕션
# test_df = pd.read_csv('./new_test_normal.csv')
# test_df= test_df.drop(['day'], axis=1)
# test_df = np.array(test_df)
# test_df = test_df.reshape(len(test_df), 1, test_df.shape[1])
# print(len(test_df))
#
# model = load_model("dnn_model.h5")
# result = model.predict(test_df, verbose=1)
#
# # (3230324, 1, 2)
# reduction = np.squeeze(result, axis=1)
# print(reduction.shape)
# print(result.shape)
# print("Submission ...")
# result_df = pd.read_csv('./dataset/sampleSubmission.csv')
# result_df.loc[:, "click"] = pd.Series(reduction[:, 0])
#
# sample_df = pd.read_csv('./dataset/sampleSubmission.csv')
# tmp_df = sample_df[len(result_df)+1:]
# result_df = pd.concat([result_df, tmp_df])
# result_df = result_df.fillna(0.5)
#
#
# result_df.to_csv("fianl_submission.csv", index=False)







# # 데이터 통계
#
# df_shuffled = csv.read_csv('./new_concated.csv').to_pandas()
#
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
#
# grp_data = pd.DataFrame(df_shuffled.groupby(['app_id', 'app_domain', 'app_category']).count())
# grp_data = grp_data.reset_index()
# grp_data.to_csv('app_info.csv')
# print(grp_data.head())
#
# grp_data = pd.DataFrame(df_shuffled.groupby(['site_id', 'site_domain', 'site_category']).count())
# grp_data = grp_data.reset_index()
# grp_data.to_csv('site_info.csv')
# print(grp_data.head())
#
# grp_data = pd.DataFrame(df_shuffled.groupby(['site_id','app_id', 'app_domain', 'site_domain']).count())
# grp_data = grp_data.reset_index()
# grp_data.to_csv('app_site_info.csv')
# print(grp_data.head())




#
# data_anal.show_how_many(df_shuffled)
#
# site_col = ["site_id", "site_domain", "site_category"]
# app_col = ["app_id", "app_domain", "app_category"]
# device_col = ["device_type"]
#
# # data_anal.show_pie(df_shuffled, site_col, "Site infos")
# # data_anal.show_pie(df_shuffled, app_col, "App infos")
# # data_anal.show_pie(df_shuffled, device_col, "Device infos")
#
# data_anal.show_num_kinds(df_shuffled, app_col)
#
# data_anal.show_hours_click(df_shuffled)
#
#
# site_kind = ['site_id', 'site_category', 'site_domain']
# app_kind = ['app_id', 'app_category', 'app_domain']
# device_kind = ['device_type']
#
# data_anal.show_clicked_info(df_shuffled, site_kind)
# data_anal.show_clicked_info(df_shuffled, app_kind)
# data_anal.show_clicked_info(df_shuffled, device_kind)
#
# print(len(df_shuffled))
#