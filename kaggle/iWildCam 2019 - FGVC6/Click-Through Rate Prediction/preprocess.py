import pandas as pd
import numpy as np
from tqdm import tqdm
from pyarrow import csv
import time


# 트레인 데이터셋 전처리
def run_preprocess(num):
    pd.set_option('display.max_columns', None)
    path = './split/split_train_' + str(num) + '.csv'
    train_df = csv.read_csv(path).to_pandas()

    # train_df = train_df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
    start = time.time()
    train_df = train_df.sample(frac=0.2, replace=True).reset_index()
    print("time :", time.time() - start)

    not_click = train_df['click'] == 0
    not_clk_df = train_df[not_click]
    not_clk_df = not_clk_df.sample(frac=0.2, replace=True)

    is_click = train_df['click'] == 1
    is_clk_df = train_df[is_click]

    print(len(not_clk_df))
    print(len(is_clk_df))

    train_df = pd.concat([not_clk_df, is_clk_df])
    train_df = train_df.reset_index()
    train_df = train_df.drop(['level_0'], axis=1)

    print(f'{path} : traing dataset preprocessing....')

    n = 10000
    list_df = [train_df[i:i+n] for i in range(0,train_df.shape[0], n)]

    targets = ['site_id','site_domain','site_category',
               'app_id','app_domain','app_category',
               'device_id','device_ip','device_model']


    #  첫번째 청크 처리
    # pre_df = drop_cols(list_df[0])
    pre_df = list_df[0].copy()
    for k in range(len(targets)):
        pre_df = hashing_info(pre_df, targets[k])
        # pre_df, label[k] = mapping_info(pre_df, targets[k], label[k])
    pre_df = split_hour(pre_df)
    list_df[0] = pre_df


    #  남은 청크들 처리 + concat
    for idx in tqdm(range(1, len(list_df))):
        # pre_df = drop_cols(list_df[idx])
        pre_df = list_df[idx].copy()
        for k in range(len(targets)):
            pre_df = hashing_info(pre_df, targets[k])

        pre_df = split_hour(pre_df[idx])
        list_df[0] = pd.concat([list_df[0], pre_df])

    train_df = list_df[0].reset_index()
    train_df = train_df.drop(['level_0'], axis=1)
    train_df = train_df.drop(['index'], axis=1)

    name = './train_preprocess/new_prepros_' + str(num) + '.csv'
    # train_df.to_csv(name, index=False)


# 테스트 데이터셋 전처리
def run_test_preprocess(num):

    pd.set_option('display.max_columns', None)
    path = './split_test/split_test_' + str(num) + '.csv'
    test_df = csv.read_csv(path).to_pandas()
    test_df = test_df.reset_index()
    print(test_df)
    print(f'{path} : test dataset preprocessing....')

    n = 10000
    list_df = [test_df[i:i+n] for i in range(0,test_df.shape[0], n)]

    targets = ['site_id','site_domain','site_category',
               'app_id','app_domain','app_category',
               'device_id','device_ip','device_model']


    #  첫번째 청크 처리
    # pre_df = drop_cols(list_df[0])
    pre_df = list_df[0].copy()
    for k in range(len(targets)):
        pre_df = hashing_info(pre_df, targets[k])
    pre_df = split_hour(pre_df)
    list_df[0] = pre_df


    #  남은 청크들 처리 + concat
    for idx in tqdm(range(1, len(list_df))):
        # pre_df = drop_cols(list_df[idx])
        pre_df = list_df[idx].copy()
        for k in range(len(targets)):
            pre_df = hashing_info(pre_df, targets[k])

        pre_df = split_hour(pre_df)
        list_df[0] = pd.concat([list_df[0], pre_df])

    test_df = list_df[0].reset_index()
    test_df = test_df.drop(['index'], axis=1)


    name = './test_preprocess/new_prepros_' + str(num) + '.csv'
    test_df.to_csv(name, index=False)



# # 컬럼 드롭
# def drop_cols(data):
#     # data = data.drop(['id', 'device_id', 'device_ip', 'device_model', 'device_conn_type', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'], axis=1)
#     data = data.drop(['id'], axis=1)
#     data = data.fillna(0)
#     data = data.astype({'hour':int, 'banner_pos':int, 'device_type':int})
#     return data
#


# hour 피쳐 split => 요일, 시간, 날짜
def split_hour(data):
    data = data.astype({'hour':int})

    day_dict = { '21': 1,
                 '22': 2,
                 '23': 3,
                 '24': 4,
                 '25': 5,
                 '26': 6,
                 '27': 0,
                 '28': 1,
                 '29': 2,
                 '30': 3,
                 '31': 4 }

    str_data = data['hour'].apply(str)


    for idx in range(data.index[0], data.index[-1] + 1):

        hour_info = str_data[idx]
        hour = int(hour_info[-2:])
        data.loc[idx, 'hour'] = hour

        date = hour_info[4:6]
        data.loc[idx, 'date'] = date

        if date in day_dict :
            date = day_dict.get(date)
            data.loc[idx, 'day'] = date


    data = data.astype({'day':int})

    return data


# 해싱 함수로 라벨 인코딩
def hashing_info(data, target):
    # print(str(data.loc[0, target]))
    for idx in range(data.index.start, data.index.stop):
        string_val = str(data.loc[idx, target])
        new_val = int(str(hash(string_val))[-6:])
        data.loc[idx, target] = new_val

    return data


# 라벨 인코딩
def mapping_info(data, target, label):

    for idx in range(len(data)):
        # print(f'{idx} / {len(data)}')
        val = data.loc[idx, target]
        if val in label:
            data.loc[idx, target] = label.index(val)
        else :
            label.append(val)
            data.loc[idx, target] = label.index(val)

    data = data.astype({target:int})

    return data, label



# 멀티프로세싱으로 전처리
def multi_calc(path_list):

    for j in range(len(path_list)):
        train_df = csv.read_csv(path_list[j]).to_pandas()

        # train_df = train_df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)
        train_df = train_df.sample(frac=0.2, replace=True).reset_index()

        not_click = train_df['click'] == 0
        not_clk_df = train_df[not_click]
        not_clk_df = not_clk_df.sample(frac=0.2, replace=True)

        is_click = train_df['click'] == 1
        is_clk_df = train_df[is_click]

        print(len(not_clk_df))
        print(len(is_clk_df))

        train_df = pd.concat([not_clk_df, is_clk_df])
        train_df = train_df.reset_index()
        train_df = train_df.drop(['level_0'], axis=1)

        print(f'{path_list[j]} : traing dataset preprocessing....')

        n = 10000
        list_df = [train_df[i:i+n] for i in range(0,train_df.shape[0], n)]

        targets = ['site_id','site_domain','site_category',
                   'app_id','app_domain','app_category',
                   'device_id','device_ip','device_model']


        #  첫번째 청크 처리
        # pre_df = drop_cols(list_df[0])
        pre_df = list_df[0].copy()
        for k in range(len(targets)):
            pre_df = hashing_info(pre_df, targets[k])
            # pre_df, label[k] = mapping_info(pre_df, targets[k], label[k])
        pre_df = split_hour(pre_df)
        list_df[0] = pre_df


        #  남은 청크들 처리 + concat
        for idx in tqdm(range(1, len(list_df))):
            # pre_df = drop_cols(list_df[idx])
            pre_df = list_df[idx].copy()
            for k in range(len(targets)):
                pre_df = hashing_info(pre_df, targets[k])

            pre_df = split_hour(pre_df)
            list_df[0] = pd.concat([list_df[0], pre_df])

        train_df = list_df[0].reset_index()
        train_df = train_df.drop(['level_0'], axis=1)
        train_df = train_df.drop(['index'], axis=1)

        # name = './train_preprocess/new_prepros_' + str(k) + '.csv'
        name = path_list[j].replace('./split/split_train_', './train_preprocess/new_prepros_')
        print(name)
        train_df.to_csv(name, index=False)


# 멀티 프로세싱 테스트 데이터셋 전처리
def multi_clac_test (path_list):

    for i in range(len(path_list)):
        test_df = csv.read_csv(path_list[i]).to_pandas()
        test_df = test_df.reset_index()

        print(f'{path_list[i]} : test dataset preprocessing....')

        n = 10000
        list_df = [test_df[i:i+n] for i in range(0,test_df.shape[0], n)]

        targets = ['site_id','site_domain','site_category',
                   'app_id','app_domain','app_category',
                   'device_id','device_ip','device_model']

        #  첫번째 청크 처리
        # pre_df = drop_cols(list_df[0])
        pre_df = list_df[0].copy()
        for k in range(len(targets)):
            pre_df = hashing_info(pre_df, targets[k])
        pre_df = split_hour(pre_df)
        list_df[0] = pre_df

        #  남은 청크들 처리 + concat
        for idx in tqdm(range(1, len(list_df))):
            # pre_df = drop_cols(list_df[idx])
            pre_df = list_df[idx].copy()
            for k in range(len(targets)):
                pre_df = hashing_info(pre_df, targets[k])

            pre_df = split_hour(pre_df)
            list_df[0] = pd.concat([list_df[0], pre_df])

        test_df = list_df[0].reset_index()
        test_df = test_df.drop(['index'], axis=1)

        name = path_list[i].replace('./split_test/split_test_', './test_preprocess/new_prepros_')
        print(name)
        test_df.to_csv(name, index=False)



#  feature 그룹별로 분할
def feature_split(data, features, train=True):
    df_sample = data[features]
    df_sample = df_sample.reset_index()
    if train :
        name = './features/feature_' + features[2] + '.csv'
    else :
        name = './test_features/feature_' + features[1] + '.csv'
    print(name)
    df_sample.to_csv(name, index=False)



# history 피쳐 생성
def bag_features(data, feature):
    # 같은 유저 정보가 여러개 있을 때, => 그룹바이해서 같은 유저 정보 뽑아내기
    same_df = pd.DataFrame(data.groupby(['id']).count())
    same_df = same_df.reset_index()

    user_list = same_df['id'].tolist()
    print(f'user list 생성 - {len(user_list)}')

    col_name = feature + '_bag'
    for i in range(len(user_list)):
        print(f'user 조회 : {i}')
        # 해당되는 유저만 조회해서 bag 에 모든 피쳐값들을 담는다.
        bag_list = []
        idx = data['id'] == user_list[i]
        user_df = data[idx]

        for k in range(len(user_df)):
            bag_list.append(data.loc[k, feature])

        bag_list.sort()

        bag_val = ("".join(map(str, bag_list)))
        bag_val = str(hash(bag_val))[-6:]


        # 해당 유저 id의 row에 bag 피쳐 추가
        for j in range(len(data)):
            if (data.loc[j, 'id'] == user_list[i]):
                data.loc[j, col_name] = bag_val

        data = data.fillna(0)
        data = data.astype({col_name:int})

        # # bag 담겼는지 체크
        # idx = data['id'] == user_list[i]
        # user_df = data[idx]
        # print(user_df)

    return data


# click history 생성
def click_history(data):
    same_df = pd.DataFrame(data.groupby(['id']).count())
    same_df = same_df.reset_index()

    user_list = same_df['id'].tolist()

    col_name = 'click_history'

    for i in range(len(user_list)):
        # 해당되는 유저만 조회해서 bag 에 모든 피쳐값들을 담는다.
        bag_list = []
        print(f'{i}번째 user {user_list[i]}')
        idx = data['id'] == user_list[i]
        user_df = data[idx].reset_index()

        for k in range(user_df.index.start, user_df.index.stop):
            bag_list.append(user_df.loc[k, 'click'])

        bag_val = ("".join(map(str, bag_list)))

        bag_val = str(hash(bag_val))[-6:]


        for j in range(len(data)):
            if (data.loc[j, 'id'] == user_list[i]):
                data.loc[j, col_name] = bag_val

        data = data.fillna(0)
        data = data.astype({col_name:int})

    return data