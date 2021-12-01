import numpy as np
import pandas as pd

import tensorflow as tf


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sys

train = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/train.csv.zip', usecols = ['Semana', 'Producto_ID', 'Demanda_uni_equil', 'Dev_uni_proxima'],)
train.head()

train_10 = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/train.csv.zip', nrows=10)

del(train)
del(train_10)

test = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/test.csv.zip')
test.head()
del(test)

train = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/train.csv.zip', nrows=5000000)
test = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/test.csv.zip', nrows=5000000)

result = train['Demanda-uni_equil'].tolist()

def label_plot(title, x, y):
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

plt.hist(result, bins=200, colot ='black')
label_plot('Distribution', 'Demanda_uni_equil', 'Count')
plt.show()

plt.hist(result, bins=50, colot ='black', range=(0,50))
label_plot('Distribution under 50', 'Demanda_uni_equil', 'Count')
plt.show()

sample = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/sample_submission.csv.zip')
sample['Demanda_uni_equil']=2
sample.to_csv('submission.csv', index=False)

train = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/train.csv.zip', nrows = 7000000)
test = pd.read_csv('/kaggle/input/grupo-bimbo-inventory-demand/test.csv.zip')

ids = test['id']
test = test.drop(['id'], axis=1)
train = train.loc[train['Demanda_uni_equil']<51, :]
test.head()
train.head()

X= train[test.columns.values]
y = train['Demanda_uni_equil']
LR = LinearRegression()
LR = LR.fit(X, y)

preds = np.around(LR.predict(test), decimals = 1)

LR_sub = pd.DataFrame({"id":ids, "Demanda_uni_equil": preds})
LR_sub.to_csv('LR_submission.csv', index=False)

def rmsle_func(truths, preds):
    truths = np.asarray(truths)
    preds = np.asarray(preds)

    n = len(truths)
    diff = (np.log(preds+1)-np.log(truths+1))**2
    return np.sqrt(np.sum(diff)/n)
rmsle = make_scorer(rmsle_func, grater_is_better=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1729)

xlf = xgb.XGBregressor(objective="reg:linear", seed=1729)
xlf.fit(X_train, y_train, eval_metric='rmse', verbose = True, eval_set = [(X_test, y_test)])

preds = np.around(xlf.predict(test), decimals=1)

xlf_sub = pd.DataFrame({"id":ids, "Demanda_uni_equil":preds})
xlf_sub.to_csv('xlf_submission.csv', index=False)