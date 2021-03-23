import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xgb

train = pd.read_csv('../208_home/data/train.csv')
test = pd.read_csv('../208_home/data/test.csv')

for i in ['year', 'month', 'day']:
    train[i] = np.nan
    test[i] = np.nan

train[['year', 'month', 'day']] = list(train.Original_Quote_Date.str.split("-"))
test[['year', 'month', 'day']] = list(test.Original_Quote_Date.str.split("-"))

train['weekday'] = pd.to_datetime(train['Original_Quote_Date']).dt.dayofweek
test['weekday'] = pd.to_datetime(test['Original_Quote_Date']).dt.dayofweek

quote_numbers = test.QuoteNumber

train.drop(['Original_Quote_Date', 'QuoteNumber'], axis=1, inplace=True)
test.drop(['Original_Quote_Date', 'QuoteNumber'], axis=1, inplace=True)

for f in train.columns:
    if train[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# %% [code]
X = train.drop('QuoteConversion_Flag', axis=1)
y = train.QuoteConversion_Flag

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

model = xgb.XGBClassifier()

xgb_model = model.fit(X_train, y_train)

output_submission = pd.DataFrame(zip(quote_numbers, xgb_model.predict_proba(test)[:, 1]),
                                 columns=['QuoteNumber', 'QuoteConversion_Flag'])
output_submission.to_csv('submission.csv', index=False)

