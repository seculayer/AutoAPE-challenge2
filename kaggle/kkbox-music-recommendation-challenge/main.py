import matplotlib as matplotlib
import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import gc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import datetime
import math


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

train = reduce_mem_usage(pd.read_csv('../input/train.csv'))
test = reduce_mem_usage(pd.read_csv('../input/test.csv'))
sei = pd.read_csv('../input/song_extra_info.csv')
members = pd.read_csv('../input/members.csv',parse_dates=['registration_init_time','expiration_date'])
songs = pd.read_csv('../input/songs.csv')

print('Shape of train is ->',train.shape)
print('Shape of test is ->',test.shape)
print('Shape of Song Extra Info is ->',sei.shape)
print('Shape of Members is ->',members.shape)
print('Shape of Songs is ->',songs.shape)

def get_codes(isrc):
    if pd.isnull(isrc):
        return np.nan
    else:
        if int(str(isrc)[5:7]) > 17:
            temp =  1900+int(str(isrc)[5:7])
        else:
            temp = 2000+int(isrc[5:7])
        return temp
    sei['year'] = sei['isrc'].apply(lambda x: get_codes(x))
    sei.sample(10)
    members['membership_days'] = members['expiration_date'].subtract(members['registration_init_time']).dt.days.astype(
        int)
    members['registration_year'] = members['registration_init_time'].dt.year
    members['expiration_year'] = members['expiration_date'].dt.year
    members.drop(columns=['registration_init_time', 'expiration_date'], inplace=True)
    members.head()
    # Extending columns
    # merging the database
    train = train.merge(songs, on='song_id', how='left')
    train = train.merge(members, on='msno', how='left')
    train = train.merge(sei, on='song_id', how='left')
    test = test.merge(songs, on='song_id', how='left')
    test = test.merge(members, on='msno', how='left')
    test = test.merge(sei, on='song_id', how='left')
    del sei, members, songs
    gc.collect()
    print(train['song_length'].isnull().value_counts() / train.shape[0])
    train['song_length'].fillna(train['song_length'].mean(), inplace=True)
    train['song_length'] = train['song_length'].astype(np.uint32)
    print(train['language'].isnull().value_counts() / train.shape[0])
    train['language'].fillna(train['language'].mode().values[0], inplace=True)
    train['language'] = train['language'].astype(np.int8)
    test['song_length'].fillna(test['song_length'].mean(), inplace=True)
    test['song_length'] = test['song_length'].astype(np.uint32)
    test['language'].fillna(test['language'].mode().values[0], inplace=True)
    test['language'] = test['language'].astype(np.int8)

    def genre_count(genre):
        if genre == 'no_genre_id':
            return 0
        else:
            return genre.count('|') + 1

    print(train['genre_ids'].isnull().value_counts() / train.shape[0])
    train['genre_ids'].fillna('no_genre_id', inplace=True)
    train['genre_ids_count'] = train['genre_ids'].apply(lambda x: genre_count(x)).astype(np.int8)
    test['genre_ids'].fillna('no_genre_id', inplace=True)
    test['genre_ids_count'] = test['genre_ids'].apply(lambda x: genre_count(x)).astype(np.int8)

    def artist_count(art):
        if art == 'no_artist_name':
            return 0
        else:
            return art.count('|') + art.count('/') + art.count('//') + art.count(';') + 1

    train['artist_name'].isnull().value_counts()
    train['artist_name'].fillna('no_artist_name', inplace=True)
    train['artist_count'] = train['artist_name'].apply(lambda x: artist_count(x)).astype(np.int8)
    test['artist_name'].fillna('no_artist_name', inplace=True)
    test['artist_count'] = test['artist_name'].apply(lambda x: artist_count(x)).astype(np.int8)

    def count_composer(comp):
        if comp == 'no_composer':
            return 0
        else:
            return comp.count('|') + comp.count('/') + comp.count('//') + comp.count(';') + 1

    def count_lyricist(lyr):
        if lyr == 'no_lyricist':
            return 0
        else:
            return lyr.count('|') + lyr.count('/') + lyr.count('//') + lyr.count(';') + 1

        train['composer'].fillna('no_composer', inplace=True)
        train['composer_count'] = train['composer'].apply(lambda x: count_composer(x)).astype(np.int8)
        train['lyricist'].fillna('no_lyricist', inplace=True)
        train['lyricist_count'] = train['lyricist'].apply(lambda x: count_lyricist(x)).astype(np.int8)
        test['composer'].fillna('no_composer', inplace=True)
        test['composer_count'] = test['composer'].apply(lambda x: count_composer(x)).astype(np.int8)
        test['lyricist'].fillna('no_lyricist', inplace=True)
        test['lyricist_count'] = test['lyricist'].apply(lambda x: count_lyricist(x)).astype(np.int8)

        dict_count_song_played_train = {k: v for k, v in train['song_id'].value_counts().iteritems()}
        dict_count_song_played_test = {k: v for k, v in test['song_id'].value_counts().iteritems()}

        def return_number_played(x):
            try:
                return dict_count_song_played_train[x]
            except KeyError:
                try:
                    return dict_count_song_played_test[x]
                except KeyError:
                    return 0

        train['number_of_time_played'] = train['song_id'].apply(lambda x: return_number_played(x))
        test['number_of_time_played'] = test['song_id'].apply(lambda x: return_number_played(x))

        dict_user_activity = {k: v for k, v in
                              pd.concat([train['msno'], test['msno']], axis=0).value_counts().iteritems()}

        def return_user_activity(x):
            try:
                return dict_user_activity[x]
            except KeyError:
                return 0

        train['user_activity_msno'] = train['msno'].apply(lambda x: return_user_activity(x))
        test['user_activity_msno'] = test['msno'].apply(lambda x: return_user_activity(x))

        # f,ax = plt.subplots(figsize=(15, 15))
        # sns.countplot(x='artist_count' ,hue= 'target'  , data = train)
        # plt.xticks(rotation=90)

        train_col = list(train.columns)
        test_col = list(test.columns)
        for f in test_col:
            if f not in train_col:
                print('ERROR !!!  Column from Test not found in train is ->', f)
        label_encoding = ['source_system_tab', 'source_screen_name',
                          'source_type', 'gender']
        drop = ['msno', 'song_id', 'isrc', 'artist_name',
                'composer', 'lyricist', 'name', 'genre_ids']
        min_max_scaling = ['number_of_time_played', 'user_activity_msno', 'membership_days', 'song_length']

        for f in label_encoding:
            lb = LabelEncoder()
            lb.fit(list(train[f].values) + list(test[f].values))
            train[f] = lb.transform(list(train[f].values))
            test[f] = lb.transform(list(test[f].values))
        for f in min_max_scaling:
            ms = MinMaxScaler()
            train[f] = ms.fit_transform(train[[f]])
            test[f] = ms.transform(test[[f]])
        # train.drop(columns = drop , inplace = True)
        # test.drop(columns=drop , inplace = True)

        for col in train.columns:
            if train[col].dtype == object:
                train[col] = train[col].astype('category')
                test[col] = test[col].astype('category')

        train.sample(10)

        X_train = train.drop(columns=['target'], axis=1)
        Y_train = train['target'].values
        X_test = test.drop(columns=['id'], axis=1)
        ids = test['id'].values
        del train, test
        gc.collect()
        train_set = lgb.Dataset(X_train, Y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting': 'gbdt',
            'learning_rate': 0.3,
            'verbose': 0,
            'num_leaves': 108,
            'bagging_fraction': 0.95,
            'bagging_freq': 1,
            'bagging_seed': 1,
            'feature_fraction': 0.9,
            'feature_fraction_seed': 1,
            'max_bin': 256,
            'max_depth': 10,
            'num_rounds': 200,
            'metric': 'auc'
        }

        #%time model_f1 = lgb.train(params, train_set=train_set,  valid_sets=train_set, verbose_eval=5)

        pred_test = model_f1.predict(X_test)
        print('Saving Predictions')
        sub = pd.DataFrame()
        sub['id'] = ids
        sub['target'] = pred_test
        sub.to_csv('1st_submission.csv', index=False, float_format='%.5f')

        sub.head()
