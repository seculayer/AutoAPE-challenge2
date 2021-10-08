import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from skimage.filters import threshold_otsu
import lightgbm as lgb
import gc
from tqdm import tqdm

SEED = 0
from warnings import filterwarnings
filterwarnings('ignore')

train = pd.read_csv("input/tabular-playground-series-sep-2021/train.csv", index_col='id')
test = pd.read_csv("input/tabular-playground-series-sep-2021/test.csv", index_col='id')

sample_submission = pd.read_csv('input/tabular-playground-series-sep-2021/sample_solution.csv')

features = [x for x in train.columns.values if x[0]=="f"]


new_features_dict = train.isna().astype(int).sum(axis=1)
new_features_test = test.isna().astype(int).sum(axis=1)
for i in range(10):
    train['miss_bt_' + str(i)] = (new_features_dict > i).astype(int)
    test['miss_bt_' + str(i)] = (new_features_test > i).astype(int)

train['n_missing'] = train[features].isna().sum(axis=1)
train['abs_sum'] = train[features].abs().sum(axis=1)
train['sem'] = train[features].sem(axis=1)
train['std'] = train[features].std(axis=1)
train['avg'] = train[features].mean(axis=1)
train['max'] = train[features].max(axis=1)
train['min'] = train[features].min(axis=1)

test['n_missing'] = test[features].isna().sum(axis=1)
test['abs_sum'] = test[features].abs().sum(axis=1)
test['sem'] = test[features].sem(axis=1)
test['std'] = test[features].std(axis=1)
test['avg'] = test[features].mean(axis=1)
test['max'] = test[features].min(axis=1)
test['min'] = test[features].min(axis=1)

fill_value_dict = {
    'f1': 'Mean',
    'f2': 'Median',
    'f3': 'Median',
    'f4': 'Median',
    'f5': 'Mode',
    'f6': 'Mean',
    'f7': 'Median',
    'f8': 'Median',
    'f9': 'Median',
    'f10': 'Median',
    'f11': 'Mean',
    'f12': 'Median',
    'f13': 'Mean',
    'f14': 'Median',
    'f15': 'Mean',
    'f16': 'Median',
    'f17': 'Median',
    'f18': 'Median',
    'f19': 'Median',
    'f20': 'Median',
    'f21': 'Median',
    'f22': 'Mean',
    'f23': 'Mode',
    'f24': 'Median',
    'f25': 'Median',
    'f26': 'Median',
    'f27': 'Median',
    'f28': 'Median',
    'f29': 'Mode',
    'f30': 'Median',
    'f31': 'Median',
    'f32': 'Median',
    'f33': 'Median',
    'f34': 'Mean',
    'f35': 'Median',
    'f36': 'Mean',
    'f37': 'Median',
    'f38': 'Median',
    'f39': 'Median',
    'f40': 'Mode',
    'f41': 'Median',
    'f42': 'Mode',
    'f43': 'Mean',
    'f44': 'Median',
    'f45': 'Median',
    'f46': 'Mean',
    'f47': 'Mode',
    'f48': 'Mean',
    'f49': 'Mode',
    'f50': 'Mode',
    'f51': 'Median',
    'f52': 'Median',
    'f53': 'Median',
    'f54': 'Mean',
    'f55': 'Mean',
    'f56': 'Mode',
    'f57': 'Mean',
    'f58': 'Median',
    'f59': 'Median',
    'f60': 'Median',
    'f61': 'Median',
    'f62': 'Median',
    'f63': 'Median',
    'f64': 'Median',
    'f65': 'Mode',
    'f66': 'Median',
    'f67': 'Median',
    'f68': 'Median',
    'f69': 'Mean',
    'f70': 'Mode',
    'f71': 'Median',
    'f72': 'Median',
    'f73': 'Median',
    'f74': 'Mode',
    'f75': 'Mode',
    'f76': 'Mean',
    'f77': 'Mode',
    'f78': 'Median',
    'f79': 'Mean',
    'f80': 'Median',
    'f81': 'Mode',
    'f82': 'Median',
    'f83': 'Mode',
    'f84': 'Median',
    'f85': 'Median',
    'f86': 'Median',
    'f87': 'Median',
    'f88': 'Median',
    'f89': 'Median',
    'f90': 'Mean',
    'f91': 'Mode',
    'f92': 'Median',
    'f93': 'Median',
    'f94': 'Median',
    'f95': 'Median',
    'f96': 'Median',
    'f97': 'Mean',
    'f98': 'Median',
    'f99': 'Median',
    'f100': 'Mode',
    'f101': 'Median',
    'f102': 'Median',
    'f103': 'Median',
    'f104': 'Median',
    'f105': 'Median',
    'f106': 'Median',
    'f107': 'Median',
    'f108': 'Median',
    'f109': 'Mode',
    'f110': 'Median',
    'f111': 'Median',
    'f112': 'Median',
    'f113': 'Mean',
    'f114': 'Median',
    'f115': 'Median',
    'f116': 'Mode',
    'f117': 'Median',
    'f118': 'Mean'
}

for col in tqdm(features):
    if fill_value_dict.get(col) == 'Mean':
        fill_value = train[col].mean()
    elif fill_value_dict.get(col) == 'Median':
        fill_value = train[col].median()
    elif fill_value_dict.get(col) == 'Mode':
        fill_value = train[col].mode().iloc[0]

    train[col].fillna(fill_value, inplace=True)
    test[col].fillna(fill_value, inplace=True)

X = train.drop(["claim"], axis=1)
X_test = test
y = train["claim"]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))
test_data = pd.DataFrame(scaler.transform(X_test))

del test, train, scaler, X_test
gc.collect()


def get_auc(y_true, y_hat):
    fpr, tpr, _ = roc_curve(y_true, y_hat)
    score = auc(fpr, tpr)
    return score


# best params
lgbm1_params = {
    'metric': 'auc',
    'max_depth': 3,
    'num_leaves': 7,
    'n_estimators': 5000,
    'colsample_bytree': 0.3,
    'subsample': 0.5,
    'random_state': 42,
    'reg_alpha': 18,
    'reg_lambda': 17,
    'learning_rate': 0.095,
    'device': 'gpu',
    'objective': 'binary'
}

lgbm2_params = {
    'metric': 'auc',
    'objective': 'binary',
    'n_estimators': 10000,
    'random_state': 42,
    'learning_rate': 0.095,
    'subsample': 0.6,
    'subsample_freq': 1,
    'colsample_bytree': 0.4,
    'reg_alpha': 10.0,
    'reg_lambda': 1e-1,
    'min_child_weight': 256,
    'min_child_samples': 20,
    'device': 'gpu',
    'max_depth': 3,
    'num_leaves': 7
}

lgbm3_params = {
    'metric': 'auc',
    'objective': 'binary',
    'device_type': 'gpu',
    'n_estimators': 10000,
    'learning_rate': 0.12230165751633416,
    'num_leaves': 1400,
    'max_depth': 8,
    'min_child_samples': 3100,
    'reg_alpha': 10,
    'reg_lambda': 65,
    'min_split_gain': 5.157818977461183,
    'subsample': 0.5,
    'subsample_freq': 1,
    'colsample_bytree': 0.2
}

catb1_params = {
    'eval_metric': 'AUC',
    'iterations': 15585,
    'objective': 'CrossEntropy',
    'bootstrap_type': 'Bernoulli',
    'od_wait': 1144,
    'learning_rate': 0.023575206684596582,
    'reg_lambda': 36.30433203563295,
    'random_strength': 43.75597655616195,
    'depth': 7,
    'min_data_in_leaf': 11,
    'leaf_estimation_iterations': 1,
    'subsample': 0.8227911142845009,
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0
}

catb2_params = {
    'eval_metric': 'AUC',
    'depth': 5,
    'grow_policy': 'SymmetricTree',
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'learning_rate': 0.1,
    'iterations': 10000,
    'loss_function': 'CrossEntropy',
    'task_type': 'GPU',
    'devices': '0',
    'verbose': 0
}

xgb1_params = {
    'eval_metric': 'auc',
    'lambda': 0.004562711234493688,
    'alpha': 7.268146704546314,
    'colsample_bytree': 0.6468987558386358,
    'colsample_bynode': 0.29113878257290376,
    'colsample_bylevel': 0.8915913499148167,
    'subsample': 0.37130229826185135,
    'learning_rate': 0.021671163563123198,
    'grow_policy': 'lossguide',
    'max_depth': 18,
    'min_child_weight': 215,
    'max_bin': 272,
    'n_estimators': 10000,
    'random_state': 0,
    'use_label_encoder': False,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor'
}

xgb2_params = dict(
    eval_metric='auc',
    max_depth=3,
    subsample=0.5,
    colsample_bytree=0.5,
    learning_rate=0.01187431306013263,
    n_estimators=10000,
    n_jobs=-1,
    use_label_encoder=False,
    objective='binary:logistic',
    tree_method='gpu_hist',
    gpu_id=0,
    predictor='gpu_predictor'
)

xgb3_params = {
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',
    'gpu_id': 0,
    'predictor': 'gpu_predictor',
    'n_estimators': 10000,
    'learning_rate': 0.01063045229441343,
    'gamma': 0.24652519525750877,
    'max_depth': 4,
    'min_child_weight': 366,
    'subsample': 0.6423040816299684,
    'colsample_bytree': 0.7751264493218339,
    'colsample_bylevel': 0.8675692743597421,
    'lambda': 0,
    'alpha': 10
}

%%time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

models = [
    ('lgbm1', LGBMClassifier(**lgbm1_params)),
    ('lgbm2', LGBMClassifier(**lgbm2_params)),
    ('lgbm3', LGBMClassifier(**lgbm3_params)),
    ('catb1', CatBoostClassifier(**catb1_params)),
    ('catb2', CatBoostClassifier(**catb2_params)),
    ('xgb1', XGBClassifier(**xgb1_params)),
    ('xgb2', XGBClassifier(**xgb2_params)),
    ('xgb3', XGBClassifier(**xgb3_params))
]

oof_pred_tmp = dict()
test_pred_tmp = dict()
scores_tmp = dict()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

for fold, (idx_train, idx_valid) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    for name, model in models:
        if name not in scores_tmp:
            oof_pred_tmp[name] = list()
            oof_pred_tmp['y_valid'] = list()
            test_pred_tmp[name] = list()
            scores_tmp[name] = list()

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=0
        )

        pred_valid = model.predict_proba(X_valid)[:, 1]
        score = get_auc(y_valid, pred_valid)

        scores_tmp[name].append(score)
        oof_pred_tmp[name].extend(pred_valid)

        print(f"Fold: {fold + 1} Model: {name} Score: {score}")
        print('--' * 20)

        y_hat = model.predict_proba(test_data)[:, 1]
        test_pred_tmp[name].append(y_hat)

    oof_pred_tmp['y_valid'].extend(y_valid)

for name, model in models:
    print(f"Overall Validation Score | {name}: {np.mean(scores_tmp[name])}")
    print('::' * 20)

base_test_predictions = pd.DataFrame(
    {name: np.mean(np.column_stack(test_pred_tmp[name]), axis=1)
    for name in test_pred_tmp.keys()}
)

base_test_predictions.to_csv('./base_test_predictions.csv', index=False)
base_test_predictions['simple_avg'] = base_test_predictions.mean(axis=1)
simple_blend_submission = sample_submission.copy()
simple_blend_submission['claim'] = base_test_predictions['simple_avg']
simple_blend_submission.to_csv('./simple_blend_submission.csv', index=False)
oof_predictions = pd.DataFrame(
    {name:oof_pred_tmp[name] for name in oof_pred_tmp.keys()}
)

oof_predictions.to_csv('./oof_predictions.csv', index=False)

y_valid = oof_predictions['y_valid'].copy()
y_hat_blend = oof_predictions.drop(columns=['y_valid']).mean(axis=1)
score = get_auc(y_valid, y_hat_blend)

print(f"Overall Validation Score | Simple Blend: {score}")
print('::'*20)

%%time
from sklearn.linear_model import LogisticRegression

X_meta = oof_predictions.drop(columns=['y_valid']).copy()
y_meta = oof_predictions['y_valid'].copy()
test_meta = base_test_predictions.drop(columns=['simple_avg']).copy()

meta_pred_tmp = []
scores_tmp = []

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

for fold, (idx_train, idx_valid) in enumerate(kf.split(X_meta, y_meta)):
    X_train, y_train = X_meta.iloc[idx_train], y_meta.iloc[idx_train]
    X_valid, y_valid = X_meta.iloc[idx_valid], y_meta.iloc[idx_valid]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    pred_valid = model.predict_proba(X_valid)[:, 1]
    score = get_auc(y_valid, pred_valid)
    scores_tmp.append(score)

    print(f"Fold: {fold + 1} Score: {score}")
    print('--' * 20)

    y_hat = model.predict_proba(test_meta)[:, 1]
    meta_pred_tmp.append(y_hat)

print(f"Overall Validation Score | Meta: {np.mean(scores_tmp)}")
print('::' * 20)

meta_predictions = np.mean(np.column_stack(meta_pred_tmp), axis=1)

stacked_submission = sample_submission.copy()
stacked_submission['claim'] = meta_predictions
stacked_submission.to_csv('./stacked_submission.csv', index=False)