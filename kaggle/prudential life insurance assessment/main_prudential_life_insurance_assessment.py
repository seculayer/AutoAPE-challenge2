import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import fmin_powell
from sklearn.metrics import cohen_kappa_score

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):

    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):

    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator


def read_csv(csv_dir):
    csv_file = pd.read_csv(csv_dir)
    #csv_file = csv_file.drop(['Insurance_History_5'],axis = 1)
    #csv_file = csv_file.drop(['Family_Hist_2'],axis = 1)
    #csv_file = csv_file.drop(['Family_Hist_3'],axis = 1)
    #csv_file = csv_file.drop(['Family_Hist_4'],axis = 1)
    #csv_file = csv_file.drop(['Family_Hist_5'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_10'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_15'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_24'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_32'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_26'],axis = 1)
    #csv_file = csv_file.drop(['Medical_History_36'],axis = 1)
    #csv_file = csv_file.drop(['Wt'],axis = 1)
    #csv_file = csv_file.drop(['Employment_Info_3'],axis = 1)
    #csv_file = csv_file.drop(['Insurance_History_4'],axis = 1)
    #csv_file = csv_file.drop(['Insurance_History_9'],axis = 1)
    print(csv_file)
    print(len(csv_file.columns))
    return csv_file


def eval_wrapper(yhat, y):
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)
    return quadratic_weighted_kappa(yhat, y)


def get_params(): #kfold 써서 조정하자
    params = {}
    params["objective"] = "reg:squarederror"  #"reg:linear"
    params["eta"] = 0.05  #0.05 ->0.1
    params["min_child_weight"] = 25   #325
    params["subsample"] = 0.8           #0.8
    params["colsample_bytree"] = 0.7    #0.7
    params["silent"] = 0                #0
    params["max_depth"] = 6          #6
    plst = list(params.items())
    print(plst)
    return plst


def score_offset(data, bin_offset, sv, scorer=eval_wrapper):# train + test 해서 confusion matrix 로 정밀도 체크
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


def apply_offsets(data, offsets):
    for j in range(num_classes):
        data[1, data[0].astype(int) == j] = data[0, data[0].astype(int) == j] + offsets[j]
    return data


# global variables
columns_to_drop = ['Id', 'Response'] #, 'Medical_History_10','Medical_History_24']
xgb_num_rounds = 720
num_classes = 8
missing_indicator = -1000


def encode_char(csv_file):
    encoder = LabelEncoder()
    encoder.fit(csv_file['Product_Info_2'])
    temp = encoder.transform(csv_file['Product_Info_2'])

    csv_file = csv_file.drop(['Product_Info_2'],axis = 1)
    csv_file['Product_Info_2'] = temp
    return csv_file

def fill_mean(csv_file):
    csv_file = csv_file.fillna(csv_file.mean(axis=1))
    csv_file = csv_file.fillna(method='ffill')
    csv_file = csv_file.fillna(method='bfill')
    return csv_file

def remove_id(csv_file):
    csv_file = csv_file.drop(['Id'],axis = 1)
    return csv_file

def remove_Response(csv_file):
    csv_file = csv_file.drop(['Response'],axis = 1)
    return csv_file


def lr(x_train,y,x_test): #x->2d, y->1d
    reg = LinearRegression()
    reg.fit(x_train,y)
    y_predict = reg.predict(x_test)
    print(y_predict)
    return y_predict

def xgboost(x_train,y,x_test):


    '''
    xgb = XGBClassifier(n_estimators=500, learning_rate=0.1,max_depth=4)
    xgb.fit(x_train,y)
    return xgb.predict(x_test)'''







def write_csv(x_test,y_predict):
    result = []
    temp = np.array(y_predict,dtype='int64')
    for i in range(len(x_test)):
        result.append([x_test[i],temp[i]])
    print(type(temp[0]))

    with open('submit.csv','w', encoding='utf-8', newline='\n') as f:
        writer = csv.writer(f)
        first = ['Id', 'Response']
        writer.writerow(first)
        for i in range(len(result)):
            writer.writerow(result[i])


if __name__ == "__main__":
    train = read_csv('../168_prudential/data/train.csv')
    test  = read_csv('../168_prudential/data/test.csv' )





    # combine train and test
    all_data = train.append(test)

    # Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
    # create any new variables
    all_data['Product_Info_2_char'] = all_data.Product_Info_2.str[0]
    all_data['Product_Info_2_num'] = all_data.Product_Info_2.str[1]

    # factorize categorical variables
    all_data['Product_Info_2'] = pd.factorize(all_data['Product_Info_2'])[0]
    all_data['Product_Info_2_char'] = pd.factorize(all_data['Product_Info_2_char'])[0]
    all_data['Product_Info_2_num'] = pd.factorize(all_data['Product_Info_2_num'])[0]

    all_data['BMI_Age'] = all_data['BMI'] * all_data['Ins_Age']

    med_keyword_columns = all_data.columns[all_data.columns.str.startswith('Medical_Keyword_')]
    all_data['Med_Keywords_Count'] = all_data[med_keyword_columns].sum(axis=1)

    print('Eliminate missing values')
    all_data.fillna(missing_indicator, inplace=True)

    # fix the dtype on the label column
    all_data['Response'] = all_data['Response'].astype(int)

    # split train and test
    train = all_data[all_data['Response'] > 0].copy()
    test = all_data[all_data['Response'] < 1].copy()

    # convert data to xgb data structure
    xgtrain = xgb.DMatrix(train.drop(columns_to_drop, axis=1), train['Response'].values,
                          missing=missing_indicator)
    xgtest = xgb.DMatrix(test.drop(columns_to_drop, axis=1), label=test['Response'].values,
                         missing=missing_indicator)

    # get the parameters for xgboost
    plst = get_params()
    print(plst)

    # train model
    model = xgb.train(plst, xgtrain, xgb_num_rounds)

    # get preds
    train_preds = model.predict(xgtrain, ntree_limit=model.best_iteration)
    print('Train score is:', eval_wrapper(train_preds, train['Response']))
    test_preds = model.predict(xgtest, ntree_limit=model.best_iteration)

    # train offsets
    offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
    offset_preds = np.vstack((train_preds, train_preds, train['Response'].values))
    offset_preds = apply_offsets(offset_preds, offsets)
    opt_order = [6, 4, 5, 3]
    for j in opt_order:
        train_offset = lambda x: -score_offset(offset_preds, x, j) * 100
        offsets[j] = fmin_powell(train_offset, offsets[j], disp=False)

    print('Offset Train score is:', eval_wrapper(offset_preds[1], train['Response']))

    # apply offsets to test
    data = np.vstack((test_preds, test_preds, test['Response'].values))
    data = apply_offsets(data, offsets)

    result = np.round(np.clip(data[1], 1, 8)).astype(int)
    write_csv(test['Id'], result)

'''
    train_x_ec = encode_char(train_x_ori)
    test_x_ec  = encode_char(test_x_ori)

    train_x_m = fill_mean(train_x_ec)
    test_x_m  = fill_mean(test_x_ec)

    train_x_id = remove_id(train_x_m)
    test_x  = remove_id(test_x_m)

    train_x = remove_Response(train_x_id)

    print(test_x.describe())
    print(train_x.describe())

    result = xgboost(train_x,train_x_ori['Response'],test_x)
    #result2 = min_max(result)'''



