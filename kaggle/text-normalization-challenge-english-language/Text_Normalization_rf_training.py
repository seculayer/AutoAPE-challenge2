import pandas as pd
import numpy as np
import os
import gc
import re
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime
from collections import Counter


train_df = pd.read_csv('./en_train.csv')
test_df = pd.read_csv('./en_test_2.csv')
train_array = train_df.values
test_array = test_df.values
gc.collect()

def make_ord(data_set):
    data_list = []
    for x in data_set:
        x_array = np.ones(20, dtype=int) * 0
        for n, i in zip(list(str(x)), np.arange(27)): x_array[i] = ord(n)             # 영단어에서 영문자 하나씩 가져옴(긴 단어가 있으므로 공간은 27개 줌)
        data_list.append(x_array)                                                      # 가져온 영문자 하나 아스키코드로 변환하고 저장
    data_x = np.array(data_list)
    return data_x

# prepare train_x
train_set = train_array[:9000000,3]
train_x = make_ord(train_set)

# prepare train_y
train_y = train_array[:9000000,2]
print(np.unique(train_y))
l_encoder = LabelEncoder()
train_y = l_encoder.fit_transform(train_y)
print(np.unique(train_y))

# prepare pred_target
before_array = test_array[:,2]
pred_target = make_ord(before_array)

print(before_array[:27])

x_train, x_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.1, random_state=1)

print("START RF!")
RF_model = RandomForestClassifier(n_estimators=100, random_state=1, min_samples_split=2, max_features=0.7,       # max_depth = 35 현재까지 최고 정확도
                                  min_samples_leaf=2, max_depth=35, n_jobs = -1).fit(x_train, y_train)

print('RF:')
print(RF_model.score(x_train, y_train))
print(RF_model.score(x_valid, y_valid))

target_label = RF_model.predict(pred_target)

label_class = l_encoder.inverse_transform(target_label)

label_class = target_label.reshape(-1,1)
before_array = before_array.reshape(-1,1)

test_set = np.concatenate((label_class, before_array), axis=1)
submission = pd.DataFrame(test_set, columns=['class','before'])
submission.to_csv("RF_upgrade.csv", index=False)