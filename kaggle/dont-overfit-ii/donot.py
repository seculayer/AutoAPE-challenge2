#python data

#数据处理和线性代数
import pandas as pd
import numpy as np

#绘图
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

#数据处理 指标 建模
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#ignore warning messages
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../input/dont-overfit-ii/train.csv')
data.head()


#Data standardizatio

#train
target_col = ["target"]
id_dataset = ["id"]
#数值列
num_cols   = [x for x in data.columns if x not in target_col + id_dataset]
#缩放数值列
std = StandardScaler()
scaled = std.fit_transform(data[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#删除原始值并合并数字列的缩放值
df_data_og = data.copy()
data = data.drop(columns = num_cols,axis = 1)
data = data.merge(scaled,left_index=True,right_index=True,how = "left")
data = data.drop(columns = ['id'],axis = 1)
#test
test = pd.read_csv('../input/dont-overfit-ii/test.csv')
target_col = ['target']
id_dataset = ['id']
#数值列
num_cols = [x for x in test.columns if x not in target_col +id_dataset]

#缩放数值列
std = StandardScaler()
scaled = std.fit_transform(test[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#删除原始值并合并数字列的缩放值
df_test_og = test.copy()
test = test.drop(columns = num_cols,axis = 1)
test = test.merge(scaled,left_index=True,right_index=True,how = "left")
test = test.drop(columns = ['id'],axis = 1)

#X and Y
X = data.drop('target',1)
y = data['target']


#Cross-validation
def scores_table(model, subtitle):
    scores = ['accuracy', 'roc_auc']
    res = []
    for sc in scores:
        scores = cross_val_score(model, X, y, cv = 5, scoring = sc)
        res.append(scores)
    df = pd.DataFrame(res).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df= df.rename(columns={0: 'accuracy', 1:'roc_auc'})

    trace = go.Table(
        header=dict(values=['<b>Fold', '<b>Accuracy','<b>Roc auc'],
                    line = dict(color='#7D7F80'),
                    fill = dict(color='#a1c3d1'),
                    align = ['center'],
                    font = dict(size = 15)),
        cells=dict(values=[('1','2','3','4','5','mean', 'std'),
                           np.round(df['accuracy'],3),
                           np.round(df['roc_auc'],3)],
                   line = dict(color='#7D7F80'),
                   fill = dict(color='#EDFAFF'),
                   align = ['center'], font = dict(size = 15)))

    layout = dict(width=800, height=400, title = '<b>Cross Validation - 5 folds</b><br>'+subtitle, font = dict(size = 15))
    fig = dict(data=[trace], layout=layout)

    py.iplot(fig, filename = 'styled_table')


#Grid search and modeling
#找到最好的超参数
random_state = 42
log_clf = LogisticRegression(random_state = random_state)
param_grid = {'class_weight' : ['balanced', None], 
                'penalty' : ['l2','l1'],  
                'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid = GridSearchCV(estimator = log_clf, param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)

grid.fit(X,y)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))

best_parameters = grid.best_params_


#Recursive feature elimination
log_clf = LogisticRegression(**best_parameters)
log_clf.fit(X,y)

selector = RFE(log_clf, 25, step=1)
selector.fit(X,y)
scores_table(selector,'selector_clf')

#submission
submission = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')
X_test = test
submission['target'] = selector.predict_proba(X_test)
submission.to_csv('submission.csv', index=False)
