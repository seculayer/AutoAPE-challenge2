import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import os
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb


def get_features_from_mjd(df):
    df['mjd_detected'] = np.NaN
    df.loc[df.detected == 1, 'mjd_detected'] = df.loc[df.detected == 1, 'mjd']
    gr_mjd = df.groupby('object_id').mjd_detected
    df['mjd_diff'] = gr_mjd.transform('max') - gr_mjd.transform('min')
    return df


def passbandSplit(df):
    df["flux_0"] = df[df.passband == 0].flux
    df["flux_1"] = df[df.passband == 1].flux
    df["flux_2"] = df[df.passband == 2].flux
    df["flux_3"] = df[df.passband == 3].flux
    df["flux_4"] = df[df.passband == 4].flux
    df["flux_5"] = df[df.passband == 5].flux

    df["abs_flux_1"] = np.abs(df["flux_1"])
    df["abs_flux_2"] = np.abs(df["flux_2"])
    df["abs_flux_3"] = np.abs(df["flux_3"])
    df["abs_flux_4"] = np.abs(df["flux_4"])
    df["abs_flux_5"] = np.abs(df["flux_5"])

    df["flux_0_err"] = df[df.passband == 0].flux_err
    df["flux_1_err"] = df[df.passband == 1].flux_err
    df["flux_4_err"] = df[df.passband == 4].flux_err
    df["flux_5_err"] = df[df.passband == 5].flux_err

    df['flux_ratio_sq_0'] = np.power(df['flux_0'] / df['flux_0_err'], 2.0)
    df['flux_ratio_sq_1'] = np.power(df['flux_1'] / df['flux_1_err'], 2.0)
    df['flux_ratio_sq_4'] = np.power(df['flux_4'] / df['flux_4_err'], 2.0)
    df['flux_ratio_sq_5'] = np.power(df['flux_5'] / df['flux_5_err'], 2.0)

    df['flux_by_flux_ratio_sq_0'] = df['flux_0'] * df['flux_ratio_sq_0']
    df['flux_by_flux_ratio_sq_1'] = df['flux_1'] * df['flux_ratio_sq_1']
    df['flux_by_flux_ratio_sq_5'] = df['flux_5'] * df['flux_ratio_sq_5']
    return df


def sabun_henkaritsu_cumsum(df):
    gr_df = df.groupby(['object_id', 'passband'])

    gr_flux = gr_df['flux']
    df["flux_henkaritsu"] = gr_flux.pct_change()

    df["flux_sabun_diff"] = gr_flux.transform('max') - gr_flux.transform('min')

    gr_mag = gr_df['magnitude']
    df["mag_sabun_diff"] = gr_mag.transform('max') - gr_mag.transform('min')
    df["mag_sabun"] = gr_mag.diff()

    gr_fl_ratio_sq = gr_df['flux_ratio_sq']
    df["fl_ratio_sabun"] = gr_fl_ratio_sq.diff()
    df["flux_detected"] = df[df.detected == 1].flux
    df["dtd_fl_by_mjd_dif"] = df["flux_detected"] / df['mjd_diff']
    df["dtd_magnitude"] = df[df.detected == 1].magnitude
    df["dtd_magnitude_1"] = df[df.detected == 1].magnitude_1
    df["dtd_magnitude_2"] = df[df.detected == 1].magnitude_2
    df["dtd_magnitude_3"] = df[df.detected == 1].magnitude_3

    gr_ob = df.groupby('object_id')
    df['dtdmag_diff'] = gr_ob.dtd_magnitude.transform('max') - gr_ob.dtd_magnitude.transform('min')
    df['dtdmag_diff_by_mjd'] = df['dtdmag_diff'] / df['mjd_diff']

    df['dtdmag_1_diff'] = gr_ob.dtd_magnitude_1.transform('max') - gr_ob.dtd_magnitude_1.transform('min')
    df['dtdmag_2_diff'] = gr_ob.dtd_magnitude_2.transform('max') - gr_ob.dtd_magnitude_2.transform('min')
    df['dtdmag_3_diff'] = gr_ob.dtd_magnitude_3.transform('max') - gr_ob.dtd_magnitude_3.transform('min')
    df['dtdmag_1_diff_by_mjd'] = df['dtdmag_1_diff'] / df['mjd_diff']
    df['dtdmag_2_diff_by_mjd'] = df['dtdmag_2_diff'] / df['mjd_diff']
    df['dtdmag_3_diff_by_mjd'] = df['dtdmag_3_diff'] / df['mjd_diff']
    del df['dtdmag_1_diff'], df['dtdmag_2_diff'], df['dtdmag_3_diff']
    df["dtdmag_pct"] = gr_ob.dtd_magnitude.pct_change()
    return df

def get_flux_decays(df):
    df['flux_1_err_diff_per_mjd'] = (df['flux_1_err_max'] - df['flux_1_err_min']) / df['mjd_diff_mean']
    del df['flux_1_err_max'], df['flux_1_err_min']

    df["flux_5_4_max"] = (df["flux_5_max"] - df["flux_4_max"])
    df["flux_4_3_max"] = (df["flux_4_max"] - df["flux_3_max"])
    df["flux_3_2_max"] = (df["flux_3_max"] - df["flux_2_max"])
    df["flux_2_1_max"] = (df["flux_2_max"] - df["flux_1_max"])
    df["flux_1_0_max"] = (df["flux_1_max"] - df["flux_0_max"])
    df["flux_5_0_mean"] = (df["flux_5_mean"] - df["flux_0_mean"])
    df["flux_3_0_mean"] = (df["flux_3_mean"] - df["flux_0_mean"])
    df["flux_4_3_std"] = (df["flux_4_std"] - df["flux_3_std"])
    df["flux_3_2_std"] = (df["flux_3_std"] - df["flux_2_std"])
    df["flux_2_1_std"] = (df["flux_2_std"] - df["flux_1_std"])
    df["flux_5_0_median"] = (df["flux_5_median"] - df["flux_0_median"])
    return df

train_set = pd.read_csv('training_set.csv', dtype={"object_id": "object"})
train_set_meta = pd.read_csv('training_set_metadata.csv', dtype={"object_id": "object"})
test_set_meta = pd.read_csv('test_set_metadata.csv')

import gc

gc.enable()

train_set = get_features_from_mjd(train_set)
train_set = passbandSplit(train_set)

train_set['flux_ratio_sq'] = np.power(train_set['flux'] / train_set['flux_err'], 2.0)
train_set['flux_by_flux_ratio_sq'] = train_set['flux'] * train_set['flux_ratio_sq']

train_set["magnitude"] = -2.5 * np.log(train_set["flux"]).fillna(0)
train_set["magnitude_0"] = -2.5 * np.log(train_set["flux_0"]).fillna(0)
train_set["magnitude_1"] = -2.5 * np.log(train_set["flux_1"]).fillna(0)
train_set["magnitude_2"] = -2.5 * np.log(train_set["flux_2"]).fillna(0)
train_set["magnitude_3"] = -2.5 * np.log(train_set["flux_3"]).fillna(0)

train_set = sabun_henkaritsu_cumsum(train_set)

aggs = {
    "dtdmag_pct": ['min', 'max', 'mean', 'median', 'std', 'skew', "sum"],
    'dtdmag_1_diff_by_mjd': ['min'],
    'dtdmag_2_diff_by_mjd': ['min'],
    'dtdmag_3_diff_by_mjd': ['min'],
    #
    'dtdmag_diff': ['min', "sum"],
    'dtdmag_diff_by_mjd': ['min', 'max', 'mean', 'median', "sum"],
    "dtd_magnitude": ['max', 'mean', 'median', 'std', 'skew'],
    "dtd_magnitude_1": ['mean', 'skew'],
    "dtd_magnitude_2": ['mean', 'std', 'skew'],
    "dtd_magnitude_3": ['std', 'skew'],
    #
    "dtd_fl_by_mjd_dif": ['max', 'std'],
    "flux_detected": ['min', 'median', 'std', 'skew'],
    #
    'magnitude': ['min', 'max', 'mean', 'std', 'skew'],
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'std'],
    'detected': ['mean'],
    'flux_ratio_sq': ['sum', 'skew'],
    'flux_by_flux_ratio_sq': ['sum', 'skew'],
    'mjd_diff': ['mean'],
    'mjd_detected': ['std'],

    'flux_0': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_1': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_2': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_3': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_4': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_5': ['min', 'max', 'mean', 'median', 'std', 'skew'],

    'abs_flux_1': ['median', 'skew'],
    'abs_flux_2': ['median', 'skew'],
    'abs_flux_3': ['skew'],
    'abs_flux_4': ['skew'],
    'abs_flux_5': ['mean', 'median'],
    'magnitude_0': ['sum'],
    'magnitude_1': ['min', 'max', 'mean'],
    'magnitude_2': ['min', 'max', 'mean'],
    'magnitude_3': ['min', 'max'],

    'flux_ratio_sq_0': ['sum'],
    'flux_ratio_sq_4': ['skew'],
    'flux_ratio_sq_5': ['sum'],
    'flux_by_flux_ratio_sq_0': ['sum'],
    'flux_by_flux_ratio_sq_1': ['skew'],
    'flux_by_flux_ratio_sq_5': ['skew'],
    'flux_1_err': ['min', 'max'],

    "flux_henkaritsu": ['min', 'median'],

    "flux_sabun_diff": ['skew'],
    "mag_sabun_diff": ['std'],
    "mag_sabun": ['min', 'std', "sum"],
    "fl_ratio_sabun": ['min', 'skew'],

}

agg_train = train_set.groupby('object_id').agg(aggs)
new_columns = [
    k + '_' + agg for k in aggs.keys() for agg in aggs[k]
]

agg_train.columns = new_columns

agg_train['magnitude_diff_by_mjd'] = (agg_train['magnitude_max'] - agg_train['magnitude_min']) / agg_train[
    'mjd_diff_mean']
agg_train['magnitude_dif2'] = (agg_train['magnitude_max'] - agg_train['magnitude_min']) / agg_train['magnitude_mean']
agg_train['flux_w_mean'] = agg_train['flux_by_flux_ratio_sq_sum'] / agg_train['flux_ratio_sq_sum']
agg_train['flux_dif3'] = (agg_train['flux_max'] - agg_train['flux_min']) / agg_train['flux_w_mean']
agg_train["flux_dif3_expo2"] = np.power(agg_train['flux_dif3'], 2)

agg_train["flux_0_median_expo2"] = np.power(agg_train['flux_0_median'], 2)
agg_train["flux_1_median_expo2"] = np.power(agg_train['flux_1_median'], 2)
agg_train["flux_5_median_expo2"] = np.power(agg_train['flux_5_median'], 2)

agg_train['magnitude_1_diff_by_mjd'] = (agg_train['magnitude_1_max'] - agg_train['magnitude_1_min']) / agg_train[
    'mjd_diff_mean']
agg_train['magnitude_2_diff_by_mjd'] = (agg_train['magnitude_2_max'] - agg_train['magnitude_2_min']) / agg_train[
    'mjd_diff_mean']
agg_train['magnitude_3_diff_by_mjd'] = (agg_train['magnitude_3_max'] - agg_train['magnitude_3_min']) / agg_train[
    'mjd_diff_mean']
del agg_train['magnitude_1_max'], agg_train['magnitude_1_min']
del agg_train['magnitude_2_max'], agg_train['magnitude_2_min']
del agg_train['magnitude_3_max'], agg_train['magnitude_3_min']

agg_train = get_flux_decays(agg_train)

del train_set
# del agg_train['flux_max']
del agg_train['flux_min'], agg_train['flux_mean'], agg_train['flux_std'], agg_train['flux_median']
del agg_train["magnitude_min"], agg_train["magnitude_max"]

print(gc.collect())
agg_train.head()

full_train = agg_train.reset_index().merge(
    right=train_set_meta, how='outer', on='object_id')

if 'target' in full_train:
    y = full_train['target']
    del full_train['target']
classes = sorted(y.unique())

class_weight = {
    c: 1 for c in classes
}
for c in [64, 15]:
    class_weight[c] = 2

print('Unique classes : ', classes)

del full_train["ra"], full_train["decl"], full_train["gal_l"], full_train["gal_b"], full_train["ddf"]
print(full_train.shape)
###
from keras.utils import to_categorical

unique_y = np.unique(y)
class_map = dict()
for i, val in enumerate(unique_y):
    class_map[val] = i

y_map = np.zeros((y.shape[0],))
y_map = np.array([class_map[val] for val in y])
y_categorical = to_categorical(y_map)


def get_luminosity_features(df):
    df["sqrd_rederr"] = np.power(df['hostgal_photoz_err'], 2.0)
    df["red_max_err"] = df["flux_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_0_max"] = df["flux_0_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_1_max"] = df["flux_1_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_2_max"] = df["flux_2_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_3_max"] = df["flux_3_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_4_max"] = df["flux_4_max"] * df["sqrd_rederr"]
    df["sqrd_rederr_5_max"] = df["flux_5_max"] * df["sqrd_rederr"]
    df["rederr_1_0_max_diff"] = df["sqrd_rederr_1_max"] - df["sqrd_rederr_0_max"]
    df["rederr_2_1_max_diff"] = df["sqrd_rederr_2_max"] - df["sqrd_rederr_1_max"]
    df["rederr_3_2_max_diff"] = df["sqrd_rederr_3_max"] - df["sqrd_rederr_2_max"]
    df["rederr_4_3_max_diff"] = df["sqrd_rederr_4_max"] - df["sqrd_rederr_3_max"]
    df["rederr_5_4_max_diff"] = df["sqrd_rederr_5_max"] - df["sqrd_rederr_4_max"]
    df["rederr_4_3_max_by"] = df["sqrd_rederr_4_max"] / df["rederr_4_3_max_diff"]
    df["rederr_3_2_max_by"] = df["sqrd_rederr_3_max"] / df["rederr_3_2_max_diff"]
    df["rederr_2_1_max_by"] = df["sqrd_rederr_2_max"] / df["rederr_2_1_max_diff"]
    df["rederr_1_0_max_by"] = df["sqrd_rederr_1_max"] / df["rederr_1_0_max_diff"]
    df["rederr__0_max_diff"] = df["red_max_err"] - df["sqrd_rederr_0_max"]
    df["rederr__1_max_diff"] = df["red_max_err"] - df["sqrd_rederr_1_max"]
    df["rederr__2_max_diff"] = df["red_max_err"] - df["sqrd_rederr_2_max"]
    del df["sqrd_rederr"]
    del df["sqrd_rederr_3_max"], df["sqrd_rederr_1_max"], df["sqrd_rederr_2_max"],
    del df["sqrd_rederr_4_max"], df["sqrd_rederr_5_max"], df["sqrd_rederr_0_max"]
    del df["rederr_1_0_max_diff"], df["rederr_4_3_max_diff"], df["rederr_3_2_max_diff"], df["rederr_2_1_max_diff"],

    ###
    df["sqrd_redshift"] = np.power(df['hostgal_photoz'], 2.0)
    df["sqrd_red_2_median"] = df["flux_2_median"] * df["sqrd_redshift"]
    df["sqrd_red_4_mean"] = df["flux_4_mean"] * df["sqrd_redshift"]
    df["sqrd_red_5_mean"] = df["flux_5_mean"] * df["sqrd_redshift"]
    df["sqrd_red_0_max"] = df["flux_0_max"] * df["sqrd_redshift"]
    df["sqrd_red_1_max"] = df["flux_1_max"] * df["sqrd_redshift"]
    df["sqrd_red_2_max"] = df["flux_2_max"] * df["sqrd_redshift"]
    df["sqrd_red_3_max"] = df["flux_3_max"] * df["sqrd_redshift"]
    df["sqrd_red_4_max"] = df["flux_4_max"] * df["sqrd_redshift"]
    df["sqrd_red_5_max"] = df["flux_5_max"] * df["sqrd_redshift"]
    df["abs_lumino_1_median"] = df["abs_flux_1_median"] * 4 * 3.14 ** df['distmod']
    df["abs_lumino_1_skew"] = df["abs_flux_1_skew"] * 4 * 3.14 ** df['distmod']
    df["abs_lumino_4_skew"] = df["abs_flux_4_skew"] * 4 * 3.14 ** df['distmod']
    ######
    df["dtd_red_min"] = df["flux_detected_min"] * df["sqrd_redshift"]
    df["dtd_red_median"] = df["flux_detected_median"] * df["sqrd_redshift"]
    df["dtd_red_std"] = df["flux_detected_std"] * df["sqrd_redshift"]
    df["dtd_red_skew"] = df["flux_detected_skew"] * df["sqrd_redshift"]
    df["dtd_red_2_median"] = df["sqrd_red_2_median"] - df["dtd_red_median"]
    df["abs_dtd_red_1_skew"] = df["abs_lumino_1_skew"] / df["dtd_red_skew"]
    df["abs_dtd_red_4_skew"] = df["abs_lumino_4_skew"] / df["dtd_red_skew"]
    ######
    df["red_max"] = df["flux_max"] * df["sqrd_redshift"]
    df["red__0_max_diff"] = df["red_max"] - df["sqrd_red_0_max"]
    df["red__1_max_diff"] = df["red_max"] - df["sqrd_red_1_max"]
    df["red__2_max_diff"] = df["red_max"] - df["sqrd_red_2_max"]
    del df["flux_max"], df["red_max"], df["sqrd_redshift"]

    df["red_2_1_max_diff"] = df["sqrd_red_2_max"] - df["sqrd_red_1_max"]
    df["red_3_2_max_diff"] = df["sqrd_red_3_max"] - df["sqrd_red_2_max"]
    df["red_5_4_max_diff"] = df["sqrd_red_5_max"] - df["sqrd_red_4_max"]
    del df["sqrd_red_3_max"]
    ###

    del df["flux_0_min"], df["flux_1_min"], df["flux_2_min"], df["flux_3_min"], df["flux_4_min"], df["flux_5_min"]
    del df["mwebv"]
    del df["flux_1_median"], df["flux_2_median"], df["flux_3_median"], df["flux_4_median"], df["flux_5_median"]
    del df["flux_3_mean"], df["flux_4_max"]
    del df['flux_ratio_sq_sum'], df['flux_by_flux_ratio_sq_sum']
    del df["flux_0_median"], df["flux_0_mean"], df["flux_0_std"]
    del df["flux_1_mean"], df["flux_1_max"],
    del df["flux_2_max"], df["flux_2_mean"], df["flux_2_std"],
    del df["flux_3_max"], df["flux_3_std"],
    del df["flux_4_std"], df["flux_4_mean"],
    del df["flux_5_max"], df["flux_5_mean"], df["flux_5_skew"]
    del df["flux_w_mean"],
    del df["magnitude_mean"],
    ##########
    return df


full_train = get_luminosity_features(full_train)
#####
del full_train['object_id'], full_train['distmod'], full_train['hostgal_specz']

train_mean = full_train.mean(axis=0)
full_train.fillna(train_mean, inplace=True)
print(full_train.shape)


def lgb_multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False


def multi_weighted_logloss(y_true, y_preds):
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss


lgb_params = {
    'device': 'cpu',
    'objective': 'multiclass',
    'num_class': 14,
    'boosting_type': 'gbdt',
    'n_jobs': -1,
    'max_depth': 7,
    'n_estimators': 500,
    'subsample_freq': 2,
    'subsample_for_bin': 5000,
    'min_data_per_group': 100,
    'max_cat_to_onehot': 4,
    'cat_l2': 1.0,
    'cat_smooth': 59.5,
    'max_cat_threshold': 32,
    'metric_freq': 10,
    'verbosity': -1,
    'metric': 'multi_logloss',
    'xgboost_dart_mode': False,
    'uniform_drop': False,
    'colsample_bytree': 0.5,
    'drop_rate': 0.173,
    'learning_rate': 0.0267,
    'max_drop': 5,
    'min_child_samples': 10,
    'min_child_weight': 100.0,
    'min_split_gain': 0.1,
    'num_leaves': 7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.00023,
    'skip_drop': 0.44,
    'subsample': 0.75
}

# Compute weights
w = y.value_counts()
weights = {i: np.sum(w) / w[i] for i in w.index}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1111)
clfs = []
importances = pd.DataFrame()
oof_preds = np.zeros((len(full_train), len(classes)))

for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
    trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
    val_x, val_y = full_train.iloc[val_], y.iloc[val_]

    clf = lgb.LGBMClassifier(**lgb_params)
    clf.fit(
        trn_x, trn_y,
        eval_set=[(trn_x, trn_y), (val_x, val_y)],
        eval_metric=lgb_multi_weighted_logloss,
        verbose=100,
        early_stopping_rounds=50,
        sample_weight=trn_y.map(weights)
    )
    oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
    print(multi_weighted_logloss(val_y, clf.predict_proba(val_x, num_iteration=clf.best_iteration_)))

    imp_df = pd.DataFrame()
    imp_df['feature'] = full_train.columns
    imp_df['gain'] = clf.feature_importances_
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)

    clfs.append(clf)

print('MULTI WEIGHTED LOG LOSS : %.5f ' % multi_weighted_logloss(y_true=y, y_preds=oof_preds))

sample_sub = pd.read_csv('sample_submission.csv')
class_names = list(sample_sub.columns[1:-1])
del sample_sub;gc.collect()

def GenUnknown(data):
    return ((((((data["mymedian"]) + (((data["mymean"]) / 2.0)))/2.0)) + (((((1.0) - (((data["mymax"]) * (((data["mymax"]) * (data["mymax"]))))))) / 2.0)))/2.0)

feats = ['class_6', 'class_15', 'class_16', 'class_42', 'class_52', 'class_53',
         'class_62', 'class_64', 'class_65', 'class_67', 'class_88', 'class_90',
         'class_92', 'class_95']

import time

start = time.time()
chunks = 5000000
for i_c, df in enumerate(pd.read_csv('test_set.csv', chunksize=chunks, iterator=True)):

    df = get_features_from_mjd(df)
    df = passbandSplit(df)

    df['flux_ratio_sq'] = np.power(df['flux'] / df['flux_err'], 2.0)
    df['flux_by_flux_ratio_sq'] = df['flux'] * df['flux_ratio_sq']

    df["magnitude"] = -2.5 * np.log(df["flux"]).fillna(0)
    df["magnitude_0"] = -2.5 * np.log(df["flux_0"]).fillna(0)
    df["magnitude_1"] = -2.5 * np.log(df["flux_1"]).fillna(0)
    df["magnitude_2"] = -2.5 * np.log(df["flux_2"]).fillna(0)
    df["magnitude_3"] = -2.5 * np.log(df["flux_3"]).fillna(0)

    df = sabun_henkaritsu_cumsum(df)

    # Group by object id
    agg_test = df.groupby('object_id').agg(aggs)
    agg_test.columns = new_columns

    agg_test['magnitude_diff_by_mjd'] = (agg_test['magnitude_max'] - agg_test['magnitude_min']) / agg_test[
        'mjd_diff_mean']
    agg_test['magnitude_dif2'] = (agg_test['magnitude_max'] - agg_test['magnitude_min']) / agg_test['magnitude_mean']
    agg_test['flux_w_mean'] = agg_test['flux_by_flux_ratio_sq_sum'] / agg_test['flux_ratio_sq_sum']
    agg_test['flux_dif3'] = (agg_test['flux_max'] - agg_test['flux_min']) / agg_test['flux_w_mean']
    agg_test["flux_dif3_expo2"] = np.power(agg_test['flux_dif3'], 2)

    agg_test["flux_0_median_expo2"] = np.power(agg_test['flux_0_median'], 2)
    agg_test["flux_1_median_expo2"] = np.power(agg_test['flux_1_median'], 2)
    agg_test["flux_5_median_expo2"] = np.power(agg_test['flux_5_median'], 2)

    agg_test['magnitude_1_diff_by_mjd'] = (agg_test['magnitude_1_max'] - agg_test['magnitude_1_min']) / agg_test[
        'mjd_diff_mean']
    agg_test['magnitude_2_diff_by_mjd'] = (agg_test['magnitude_2_max'] - agg_test['magnitude_2_min']) / agg_test[
        'mjd_diff_mean']
    agg_test['magnitude_3_diff_by_mjd'] = (agg_test['magnitude_3_max'] - agg_test['magnitude_3_min']) / agg_test[
        'mjd_diff_mean']
    del agg_test['magnitude_1_max'], agg_test['magnitude_1_min']
    del agg_test['magnitude_2_max'], agg_test['magnitude_2_min']
    del agg_test['magnitude_3_max'], agg_test['magnitude_3_min']

    agg_test = get_flux_decays(agg_test)
    #

    #    del agg_test['flux_max'],
    del agg_test['flux_min'], agg_test['flux_mean'], agg_test['flux_std'], agg_test['flux_median']
    del agg_test["magnitude_min"], agg_test["magnitude_max"]

    # Merge with meta data
    full_test = agg_test.reset_index().merge(
        right=test_set_meta, how='left', on='object_id')

    del full_test["ra"], full_test["decl"], full_test["gal_l"], full_test["gal_b"], full_test["ddf"]

    full_test = get_luminosity_features(full_test)
    full_test = full_test.fillna(train_mean)

    # Make predictions
    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(full_test[full_train.columns]) / folds.n_splits
        else:
            preds += clf.predict_proba(full_test[full_train.columns]) / folds.n_splits

    # Compute preds_99 as the proba of class not being any of the others
    # preds_99 = 0.1 gives 1.769
    preds_99 = np.ones(preds.shape[0])
    for i in range(preds.shape[1]):
        preds_99 *= (1 - preds[:, i])

    # Store predictions
    preds_df = pd.DataFrame(preds, columns=['class_' + str(s) for s in clfs[0].classes_])
    preds_df['object_id'] = full_test['object_id']
    preds_df['class_99'] = 0.14 * preds_99 / np.mean(preds_99)

    ##########
    y = pd.DataFrame()
    y['mymean'] = preds_df[feats].mean(axis=1)
    y['mymedian'] = preds_df[feats].median(axis=1)
    y['mymax'] = preds_df[feats].max(axis=1)

    preds_df['class_99'] = GenUnknown(y)
    ##########

    if i_c == 0:
        preds_df.to_csv('predictions.csv', header=True, mode='a', index=False)
    else:
        preds_df.to_csv('predictions.csv', header=False, mode='a', index=False)

    del agg_test, full_test, preds_df, preds
    gc.collect()

    if (i_c + 1) % 10 == 0:
        print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
z = pd.read_csv('predictions.csv')

print(z.groupby('object_id').size().max())
print((z.groupby('object_id').size() > 1).sum())

z = z.groupby('object_id').mean()

z.to_csv('great_trust_cv_predictions.csv', index=True)