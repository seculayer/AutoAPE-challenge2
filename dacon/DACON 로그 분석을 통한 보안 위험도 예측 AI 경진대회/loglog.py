# # # # # # # # # # random forest # # # # # # # # # # 
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

warnings.filterwarnings(action='ignore')
pd.set_option('display.max_colwidth', None)


# # # 데이터 로드
train=pd.read_csv('./loglog/train.csv')
test=pd.read_csv('./loglog/test.csv')
submission=pd.read_csv('./loglog/sample_submission.csv')

# train dataset id 컬럼 제거
train = train.drop('id', axis = 1)

# train level별 값 확인
print(train['level'].value_counts())


# # # 데이터 전처리
# 1. 날짜, 시간, 아이파, 포트, 한글, 숫자 등 마스킹
# 2. 특수문자 제거
# 3. 소문자로 변환
print(train.head(5))

train['full_log'] = train['full_log'].str.replace(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?:[0-9]| [0-9]|[1-3][0-9]) (?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<MONTHDAYTIME>')
train['full_log'] = train['full_log'].str.replace(r'\d{4}\-\d{2}\-\d{2}T(?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<DATETIME>')
train['full_log'] = train['full_log'].str.replace(r'(?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<TIME>')
train['full_log'] = train['full_log'].str.replace(r'(?:[01][0-9]|2[0-3])[:h][0-5][0-9]', '<TIME>')

train['full_log'] = train['full_log'].str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<IPADDRESS>')
train['full_log'] = train['full_log'].str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IPADDRESS>')
train['full_log'] = train['full_log'].str.replace(r'port \d+', '<PORT>')

train['full_log'] = train['full_log'].str.replace(r'[ㄱ-ㅣ가-힣]+', '<KOREAN>').str.replace(r'[0-9]+', '<NUM>')

train['full_log'] = train['full_log'].str.replace(r'[^a-zA-Z<>]', ' ')
train['full_log'] = train['full_log'].str.lower()

print(train.head(5))

# 4. 중복제거
drop_train = train.drop_duplicates()
x = drop_train[drop_train.duplicated(['full_log'], keep=False)].drop_duplicates(subset = ['full_log'])['full_log']

for i in x:
    
    v = train[train['full_log'] == i]['level'].mode().values
    
    if len(v) == 1:
        train.loc[(train['full_log'] == i), 'level'] = v[0]
        
    else:
        fvc = 0
        for j in train[train['full_log'] == i]['level'].unique():
            vc = train['level'].value_counts().loc[j]
            if vc > fvc :
                fvc = vc
                m = j
            else :
                pass
        train.loc[(train['full_log'] == i), 'level'] = m

print(train['level'].value_counts())

# train full_log 전처리와 같은 방식으로 전처리 
# full_log에서 날짜, 시간, IP, 한글, 숫자를 마스킹하고 특수문자 제거 및 소문자 변환
test['full_log'] = test['full_log'].str.replace(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?:[0-9]| [0-9]|[1-3][0-9]) (?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<MONTHDAYTIME>')
test['full_log'] = test['full_log'].str.replace(r'\d{4}\-\d{2}\-\d{2}T(?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<DATETIME>')
test['full_log'] = test['full_log'].str.replace(r'(?:[01][0-9]|2[0-3])[:h][0-5][0-9][:m][0-5][0-9]', '<TIME>')
test['full_log'] = test['full_log'].str.replace(r'(?:[01][0-9]|2[0-3])[:h][0-5][0-9]', '<TIME>')

test['full_log'] = test['full_log'].str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d+', '<IPADDRESS>')
test['full_log'] = test['full_log'].str.replace(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '<IPADDRESS>')
test['full_log'] = test['full_log'].str.replace(r'port \d+', '<PORT>')

test['full_log'] = test['full_log'].str.replace(r'[ㄱ-ㅣ가-힣]+', '<KOREAN>').str.replace(r'[0-9]+', '<NUM>')

test['full_log'] = test['full_log'].str.replace(r'[^a-zA-Z<>]', ' ')
test['full_log'] = test['full_log'].str.lower()

# train['full_log'] => train_text로 list
# train['level']=> train_level로 array
train_text=list(train['full_log'])
train_level=np.array(train['level'])

# CountVectorizer로 벡터화
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(analyzer="word", max_features=10000)

train_features=vectorizer.fit_transform(train_text)
print(train_features.shape)


# # # 모델링
#훈련 데이터 셋과 검증 데이터 셋으로 분리
TEST_SIZE = 0.3
RANDOM_SEED = 666
train_x, eval_x, train_y, eval_y = train_test_split(train_features, train_level, test_size=TEST_SIZE, random_state=RANDOM_SEED)

#랜덤포레스트로 모델링
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(n_estimators=100, random_state = RANDOM_SEED, class_weight = 'balanced')
forest.fit(train_x, train_y)

#모델 검증
print(forest.score(eval_x, eval_y))

# crosstab으로 확인
pred=forest.predict(eval_x)
crosstab = pd.crosstab(eval_y, pred, rownames=['real'], colnames=['pred'])
print(crosstab)

# 새로운 위험요소에 대한 가정 추가
preds=forest.predict(eval_x)
probas=forest.predict_proba(eval_x)
print(preds.shape)
print(probas.shape)

preds[np.where(np.max(probas, axis=1) < 0.7)]=7
new_crosstab = pd.crosstab(eval_y, preds, rownames=['real'], colnames=['pred'])
print(new_crosstab)


# # # 예측
# test['full_log'] => test_text로 list
test_text=list(test['full_log'])
ids=list(test['id'])

# test 데이터 벡터화
test_features = vectorizer.transform(test_text)

results=forest.predict(test_features)
results_proba=forest.predict_proba(test_features)

results_new = results.copy()
results_new[np.where((np.max(results_proba, axis=1) < 0.7) & (results != 6) & (results != 4) & (results != 2))] = 7

# level이 2,4,6이 아니고 예측치의 예측 확률이 0.7 이하, 즉 확신이 없을 경우 이상치로 판단
# ** level 2,4,6의 경우 값이 적어서 일단 이상치 판단에서 제외
submission['level']=results_new

print(submission['level'].value_counts())

# submission 저장
submission.to_csv('./loglog/log_output_3.csv', index=False)
print(submission)


