import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from tqdm import tqdm
import time
from sklearn.metrics import roc_auc_score

import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
import shap
import optuna

# 데이터 로드
data_dir = './marchi'
train = pd.read_csv(data_dir + '/train.csv', index_col='index')
test = pd.read_csv(data_dir + '/test_x.csv', index_col='index')
submission = pd.read_csv(data_dir + '/sample_submission.csv', index_col='index')

# 레이블 분리 / 테스트 세트 복제
Ytrain = train['voted']
Xtrain = train.drop(columns='voted')
Xtest = test.copy()

print(Xtrain.shape, Ytrain.shape, Xtest.shape, submission.shape)


feature_names=[]
model = LGBMClassifier()
skf = StratifiedKFold(n_splits=5)

# 스코어룰 정의
base_score = cross_val_score(model, pd.get_dummies(Xtrain, columns = list(Xtrain.select_dtypes(np.object))), Ytrain, cv=skf, 
                             scoring='roc_auc').mean()
print(f"Doing Nothing : {base_score}")


all_data = pd.concat((Xtrain, Xtest), axis=0)
print(all_data.shape)


# 선택 컬럼들 수치화 / 레이블 인코딩 (원한인코딩과 같다)
all_data = pd.get_dummies(all_data, columns=['race','religion','married','hand','gender','engnat','age_group'])
le = LabelEncoder()
for col in ['education','urban']:
    all_data[col] = le.fit_transform(all_data[col])
print(cross_val_score(model, all_data[:len(Ytrain)], Ytrain, cv=skf, scoring='roc_auc').mean())

# all_data 원핫인코딩 해주고 다시 학습이랑 테스트 세트 분할
Xtrain = all_data[:len(Ytrain)]
Xtest = all_data[len(Ytrain):]
print(Xtrain.shape, Ytrain.shape, Xtest.shape)

# 스케일링(정규화)
scaler = MinMaxScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)
Ytrain_m1 = Ytrain - 1


# 앙상블 대상인 7가지 모델의 파라미터 튠 함수
def tune_model(model_type, X, Y, n_trials=100, cv=5):
    skf = StratifiedKFold(n_splits=cv)
    scaler = MinMaxScaler()
    
    if model_type not in ['lgb','xgb','cat','rf','ets','logistic','knn']:
        raise ValueError('Only Support [ "lgb", "xgb", "cat", "rf", "ets", "logistic", "knn" ] model_type')
    
    # lgbm parameter set
    if model_type=='lgb':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            num_leaves = trial.suggest_int('num_leaves', 7, 255)
            reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 100)
            reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 100)
            colsample_bytree = trial.suggest_uniform('colsample_bytree', 0, 1)
            subsample = trial.suggest_uniform('subsample', 0, 1)
            params = {'n_estimators' : n_estimators, 
                     'learning_rate' : learning_rate, 
                      'num_leaves' : num_leaves,
                     'reg_alpha' : reg_alpha,
                     'reg_lambda' : reg_lambda, 
                     'colsample_bytree' : colsample_bytree, 
                     'subsample' : subsample
                     }
            model = LGBMClassifier(metric='auc', subsample_freq=1, random_state=18, **params)
            score = cross_val_score(model, X, Y, cv=skf, scoring='roc_auc').mean()
            return score
        
    # xgboost parameter set 정의
    elif model_type=='xgb':    
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 8)
            reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-4, 100)
            reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-4, 100)
            colsample_bytree = trial.suggest_uniform('colsample_bytree', 0, 1)
            subsample = trial.suggest_uniform('subsample', 0, 1)
            params = {'n_estimators' : n_estimators, 
                     'learning_rate' : learning_rate, 
                      'max_depth' : max_depth,
                     'reg_alpha' : reg_alpha,
                     'reg_lambda' : reg_lambda, 
                     'colsample_bytree' : colsample_bytree, 
                     'subsample' : subsample
                     }
            model = XGBClassifier(eval_metric='auc', random_state=18, **params, verbosity=0)   # gpu 가속 : tree_method='gpu_hist', 
            score = cross_val_score(model, X, Y, cv=skf, scoring='roc_auc').mean()
            return score
    
    # catboost parameter set 정의
    elif model_type=='cat':
        def objective(trial):
            params={}
            params['n_estimators'] = trial.suggest_int('n_estimators', 100, 1000)
            params['learning_rate'] = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            params['depth'] = trial.suggest_int('depth', 3, 8)
            params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-4, 30)
            params['random_strength'] = trial.suggest_uniform('random_strength', 0.1, 30)
            
            params['bootstrap_type'] = trial.suggest_categorical('bootstrap_type', ['Bayesian','Bernoulli','Poisson'])
            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_uniform('bagging_temperature', 0, 30)
            else: 
                params['subsample'] = trial.suggest_uniform('subsample', 0, 1)
            print(params)
            model = CatBoostClassifier(eval_metric='AUC', random_seed=18, **params, verbose=False) # gpu 가속 : task_type='GPU', 
            score = cross_val_score(model, X, Y, cv=skf, scoring='roc_auc').mean()
            return score

    # random forest parameter set 정의
    elif model_type=='rf':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 100)
            max_features = trial.suggest_uniform('max_features', 0, 1)
            criterion = trial.suggest_categorical('criterion', ['gini','entropy'])
            params = {'n_estimators' : n_estimators, 
                     'max_features' : max_features, 
                      'max_depth' : max_depth,
                      'criterion' : criterion
                     }
            model = RandomForestClassifier(random_state=18, **params, n_jobs=-1)
            score = cross_val_score(model, X, Y, cv=skf, scoring='roc_auc').mean()
            return score
    
    # extra trees parameter set 정의
    elif model_type=='ets':
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 100)
            max_features = trial.suggest_uniform('max_features', 0, 1)
            criterion = trial.suggest_categorical('criterion', ['gini','entropy'])
            params = {'n_estimators' : n_estimators, 
                     'max_features' : max_features, 
                      'max_depth' : max_depth,
                      'criterion' : criterion
                     }
            model = ExtraTreesClassifier(random_state=18, **params, n_jobs=-1)
            score = cross_val_score(model, X, Y, cv=skf, scoring='roc_auc').mean()
            return score
    
    # logistic regressor parameter set 정의
    elif model_type=='logistic':
        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-4, 100)
            solver = trial.suggest_categorical('solver', ['lbfgs','sag','saga','liblinear'])
            model = LogisticRegression(verbose=False, n_jobs=-1, warm_start=False, random_state=18, max_iter=2000, 
                                      C=C, solver=solver)
            score = cross_val_score(model, scaler.fit_transform(X), Y, cv=skf, scoring='roc_auc').mean()
            return score
    
    # kneighbor parameter set 정의
    elif model_type=='knn':
        def objective(trial):
            n_neighbors = trial.suggest_int('n_neighbors', 1, 2048)
            p = trial.suggest_uniform('p', 1, 10)
            leaf_size = trial.suggest_int('leaf_size', 10, 300)
            
            model = KNeighborsClassifier(n_jobs=-1, n_neighbors=n_neighbors, p=p, leaf_size=leaf_size, weights='distance')
            score = cross_val_score(model, scaler.fit_transform(X), Y, cv=skf, scoring='roc_auc').mean()
            return score

    # optuna로 하이퍼파라미터 자동 튜닝
    # direction : 스코어를 최대 혹은 최소로 하는 방향을 지정
    # 최적의 파라미터를 뱉어줌
    study=optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    print(f"Model : {model_type}, Best Score : {study.best_value}, Best Params : {study.best_params}")
    return study.best_params

stack_train = pd.DataFrame(data=None, index=train.index)
stack_test = pd.DataFrame(data=None, index=test.index)
skf = StratifiedKFold(n_splits=10)


# catboost training
cat_params = {'n_estimators': 890, 'learning_rate': 0.03222314337959399, 'depth': 5, 'reg_lambda': 26.07229261551145, 
              'random_strength': 7.099137266222318, 'bootstrap_type': 'Bernoulli', 'subsample': 0.8777001157409003}
model_cat = CatBoostClassifier(eval_metric='AUC', verbose=False, **cat_params, random_seed=18) # gpu 가속 : task_type='GPU',

stack_train['CAT'] = cross_val_predict(model_cat, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_cat.fit(Xtrain, Ytrain)
stack_test['CAT'] = model_cat.predict_proba(Xtest)[:,1]

# xgboost training
xgb_params = tune_model('xgb', Xtrain, Ytrain, n_trials=100, cv=10)
model_xgb = XGBClassifier(eval_metric='auc', verbosity=0, **xgb_params, random_state=18) # gpu 가속 : tree_method='gpu_hist', 
stack_train['XGB'] = cross_val_predict(model_xgb, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_xgb.fit(Xtrain, Ytrain)
stack_test['XGB'] = model_xgb.predict_proba(Xtest)[:,1]

# lgbm training
lgb_params = tune_model('lgb', Xtrain, Ytrain, n_trials=100, cv=10)
model_lgb = LGBMClassifier(subsample_freq=1, random_state=18, **lgb_params)
stack_train['LGB'] = cross_val_predict(model_lgb, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_lgb.fit(Xtrain, Ytrain)
stack_test['LGB'] = model_lgb.predict_proba(Xtest)[:,1]

# rf training
rf_params = tune_model('rf', Xtrain, Ytrain, n_trials=100, cv=10)
model_rf = RandomForestClassifier(n_jobs=-1, random_state=18, **rf_params)
stack_train['RF'] = cross_val_predict(model_rf, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_rf.fit(Xtrain, Ytrain)
stack_test['RF'] = model_rf.predict_proba(Xtest)[:,1]

# extra tree training
ets_params = tune_model('ets', Xtrain, Ytrain, n_trials=100, cv=10)
model_ets = ExtraTreesClassifier(n_jobs=-1, random_state=18, **ets_params)
stack_train['ETS'] = cross_val_predict(model_ets, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_ets.fit(Xtrain, Ytrain)
stack_test['ETS'] = model_ets.predict_proba(Xtest)[:,1]

# logistic training
logistic_params = tune_model('logistic', Xtrain, Ytrain, n_trials=100, cv=10)
model_logistic = LogisticRegression(verbose=True, n_jobs=-1, warm_start=True, random_state=18, max_iter=2000, **logistic_params)
stack_train['LOGISTIC'] = cross_val_predict(model_logistic, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_logistic.fit(Xtrain, Ytrain)
stack_test['LOGISTIC'] = model_logistic.predict_proba(Xtest)[:,1]

# knn training
knn_params = tune_model('knn', Xtrain, Ytrain, n_trials=100, cv=10)
model_knn = KNeighborsClassifier(n_jobs=-1, **knn_params, weights='distance')
stack_train['KNN'] = cross_val_predict(model_knn, Xtrain, Ytrain, method='predict_proba', cv=skf)[:,1]
model_knn.fit(Xtrain, Ytrain)
stack_test['KNN'] = model_knn.predict_proba(Xtest)[:,1]


# 위에서 진행한 여섯개의 모델의 가중치를 평균내주고 softmax out
# Weighted Average
from scipy.special import softmax
def objective_weight(trial):
    weights=[]
    for i in range(len(list(stack_train))):
        weights.append(trial.suggest_uniform(f"weight_{i}", 0, 100))
    weights = softmax(weights)
    
    weighted_average = np.zeros(shape=Ytrain.shape)
    for col, weight in zip(stack_train.columns, weights):
        weighted_average += stack_train[col] * weight
        
    score = roc_auc_score(Ytrain, weighted_average)
    return score

# 위에서 7가지 모델의 자동 튜닝을 했으면 앙상블된 통합 모델의 튜닝을 진행함
# 뱉는건 softmax로 뱉어준다
study_weight = optuna.create_study(direction='maximize')
study_weight.optimize(objective_weight, n_trials=1000)
print(softmax(list(study_weight.best_params.values())), study_weight.best_value)

weights = softmax(list(study_weight.best_params.values()))
print(weights)

# stack_test 변수에 저장되어 있는 예측값에 컬럼별로 가중치를 곱해서 값을 구하고
# 걔네를 다 더해서 row 별 평균 가중치를 구해서 weighted_average 배열에 집어 넣어줌
# 표현 알아둘것. 유용해보임
weighted_average = np.zeros(shape = submission.shape).reshape(-1)
for col, weight in zip(list(stack_test), weights):
    value = stack_test[col] * weight
    weighted_average += value
submission['voted'] = weighted_average

# 결과 뱉뱉
rank_avg = submission.copy()
rank_avg = rank_avg.sort_values(by='voted')
rank_avg['rank'] = np.arange(0, len(rank_avg))
rank_avg = rank_avg.sort_values(by='index')
submission['voted'] = rank_avg.mean(axis=1)
submission['voted'] = scaler.fit_transform(submission)
submission.to_csv(data_dir + '/marchi_ensemble_output.csv')
print(submission)


# 결과 확인해보고 좋지 않을 경우 원본의 keras MLP 추가해서 돌려볼것.



