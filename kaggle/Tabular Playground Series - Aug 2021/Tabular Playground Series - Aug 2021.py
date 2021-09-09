# !pip install -U lightautoml

import pandas as pd

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

N_THREADS = 4
N_FOLDS = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
TIMEOUT = 3*3600
TARGET_NAME = 'loss'

train_data = pd.read_csv('input/tabular-playground-series-aug-2021/train.csv')
print(train_data.head())

test_data = pd.read_csv('input/tabular-playground-series-aug-2021/test.csv')
print(test_data.head())

samp_sub = pd.read_csv('input/tabular-playground-series-aug-2021/sample_submission.csv')
print(samp_sub.head())


lgb_params = {
    'metric': 'RMSE',
    'lambda_l1': 1e-07,
    'lambda_l2': 1e-07,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'min_child_samples': 19,
    'num_threads': 4
}

cb_params = {
    'num_trees': 1000,
    'od_wait': 1200,
    'learning_rate': 0.1,
    'l2_leaf_reg': 64,
    'subsample': 0.83,
    'random_strength': 17.17,
    'max_depth': 16,
    'min_data_in_leaf': 20,
    'leaf_estimation_iterations': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'bootstrap_type': 'Bernoulli',
    'leaf_estimation_method': 'Newton',
    'random_seed': 42,
    "thread_count": 4
}

automl = TabularAutoML(task = Task('reg', ),
                       timeout = TIMEOUT,
                       cpu_limit = N_THREADS,
                       reader_params = {'n_jobs': N_THREADS, 'cv': N_FOLDS, 'random_state': RANDOM_STATE},
                       general_params = {'use_algos': [['lgb', 'cb']]},
                       lgb_params = {'default_params': lgb_params, 'freeze_defaults': True},
                       cb_params = {'default_params': cb_params, 'freeze_defaults': True},
                       verbose = 2
                      )

pred = automl.fit_predict(train_data, roles = {'target': TARGET_NAME, 'drop': ['id']})

test_pred = automl.predict(test_data)

samp_sub['loss'] = test_pred.data[:, 0]
samp_sub.to_csv('submission.csv', index = False)