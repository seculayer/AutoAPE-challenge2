import sys
!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz
!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null
sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path
sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 
!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import math
import sys

import numpy as np
import pandas as pd
import gc
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import cudf
from cuml.ensemble import RandomForestRegressor

#read data ”
df      = pd.read_csv("../input/remove-trends-giba-explained/train_clean_giba.csv").sort_values("time").reset_index(drop=True)
test_df = pd.read_csv("../input/remove-trends-giba-explained/test_clean_giba.csv").sort_values("time").reset_index(drop=True)

df.signal        = df.signal.astype('float32')
df.open_channels = df.open_channels.astype('float32')

test_df.signal   = test_df.signal.astype('float32')




# train segments with more then 9 open channels classes
# test segments with more then 9 open channels classes (potentially)

df["category"] = 0
test_df["category"] = 0

df.loc[2_000_000:2_500_000-1, 'category'] = 1
df.loc[4_500_000:5_000_000-1, 'category'] = 1

test_df.loc[500_000:600_000-1, "category"] = 1
test_df.loc[700_000:800_000-1, "category"] = 1

df['category']      = df['category'].astype( np.float32 )
test_df['category'] = test_df['category'].astype( np.float32 )

TARGET = "open_channels"

aug_df = df[df["group"] == 5].copy()
aug_df["category"] = 1
aug_df["group"] = 10

for col in ["signal", TARGET]:
    aug_df[col] += df[df["group"] == 8][col].values
    
aug_df['category'] = aug_df['category'].astype( np.float32 )
    

df = df.append(aug_df, sort=False)
NUM_SHIFT = 20

features = ["signal","signal","category"]

for i in range(1, NUM_SHIFT + 1):
    f_pos = "shift_{}".format(i)
    f_neg = "shift_{}".format(-i)
    features.append(f_pos)
    features.append(f_neg)
    for data in [df, test_df]:
        data[f_pos] = data["signal"].shift(i).fillna(-3).astype( np.float32 ) # Groupby shift!!!
        data[f_neg] = data["signal"].shift(-i).fillna(-3).astype( np.float32 ) # Groupby shift!!!
        
data.head()


# RandomForestRegressor mode
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

test_df = cudf.from_pandas( test_df )

oof_preds = np.zeros((len(df)))
y_test = np.zeros((len(test_df)))
for fold, (train_ind, val_ind) in enumerate(skf.split(df, df["group"])):
    train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]
    print('Fold', fold )

    train_df = cudf.from_pandas( train_df )
    val_df   = cudf.from_pandas( val_df )

    model = RandomForestRegressor(
            n_estimators=35,
            rows_sample = 0.35,
            max_depth=18,
            max_features=11,        
            split_algo=0,
            bootstrap=False,
        ).fit( train_df[features], train_df[TARGET] )
        
    pred = model.predict( val_df[features] ).to_array()
    oof_preds[val_ind] = np.round( pred )
        
    y_test += model.predict( test_df[features] ).to_array() / NUM_FOLDS
    del model; _=gc.collect()
    
y_test = np.round( y_test )

#Create Submission
f1_score( df["open_channels"], oof_preds, average="macro")
test_df['time'] = [ "{0:.4f}".format(v) for v in test_df['time'].to_array() ]
test_df[TARGET] = y_test.astype(np.int32)
test_df.to_csv('submission.csv', index=False, columns=["time", TARGET])
