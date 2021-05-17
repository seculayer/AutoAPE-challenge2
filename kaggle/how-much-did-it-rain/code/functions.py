import pandas as pd
import numpy as np
import sys
sys.path.append('xgboost-0.40/wrapper/')
import xgboost as xgb

# Adapted from CRPS calculation code by Alexander Guschin as posted on kaggle forums
# Provides some additional useful debugging information if the format is incorrect.
def is_cdf_valid(case):
    #case= np.around(case,6)
    if case[0] < 0 or case[0] > 1:
        return False
    for i in xrange(1, len(case)):
        if case[i] > 1 or case[i] < case[i-1]:
            print('case[i] = ', case[i])
            print 'case[i] > 1 = ', case[i] > 1
            print 'case[i] < case[i-1] = ', case[i] < case[i-1]
            return False
    return True

#calculates heaviside in one step.  
def calc_crps(thresholds,predictions, actuals):
    obscdf = (thresholds.reshape(70,1) >= actuals).T
    crps = np.mean(np.mean((predictions - obscdf) ** 2))
    return crps

#for validating final solutions without any actual labels
def check_solution(predictions):
    for e,p in enumerate(predictions) :
        if is_cdf_valid(p) == False : 
            print 'something wrong with your prediction'
            print e 
            break
    return 0

###############################################################################################

#combines labels together as specficied by label_list
#label list is a nested list of form [[[1,2,3],1],...] original label values 1,2, and 3 will be replaced by 1
def aggregate_labels(label_list, ceiled_labels):
    new_lab = ceiled_labels.replace(label_list[0][0],label_list[0][1])    
    for i in range(1,len(label_list)):
        new_lab = new_lab.replace(label_list[i][0],label_list[i][1])
    return new_lab

#takes the ceiling of the actual labels, aggregates values above the cutoff
def discretize_labels(labels,cutoff=69):
    new_labels = np.zeros(len(labels.index))
    for e,i in enumerate(labels.values.ravel()):
        if i > cutoff:
            new_labels[e] = 70
        elif i <= cutoff and i >= cutoff:
            new_labels[e] = cutoff
        else:
            new_labels[e] = np.ceil(i)
    return pd.DataFrame(np.abs(new_labels))


#returns a numpy array of format [0,0,0,1,1,1...], where the transition from zero to one occurs 
# at the actual label value, the above case the label would be 4mm
def make_cdf_step(true_label_value):
    step_cdf = np.ones(70)
    step_cdf[0:true_label_value] = 0
    return step_cdf

#return empirical cumalitive distribution for labels in an aggregate
def make_cdf_distribution(in_class_labels):
    pdf = in_class_labels.value_counts()/float(len(in_class_labels))
    pdf = pdf.sort_index()
    cdf = np.zeros(70)
    for e,i in enumerate(pdf.index.values.tolist()):
        cdf[i] = pdf.iloc[e]
    return cdf.cumsum()

#returns a list of cdf. 
def make_cdf_list(first_agg, num_lab, new_lab, actual_labels,offset):
    cdfs = []
    for i in range(num_lab):
        if i < first_agg: 
            cdfs.append(make_cdf_step(i))
        else:
            cdfs.append(make_cdf_distribution(actual_labels.reindex(new_lab.iloc[offset:][new_lab.iloc[offset:]==i].index)))
    return cdfs

# Takes a list of partial CDFs and the output of xgboost and combines them to create the final 
# this function assumes that zero is always it's own class. 
def create_full_predictions(CDFs, predictions):
    data_length = len(predictions)
    for e,i in enumerate(CDFs):
        if len(i.shape) == 1:
            if e == 0:
                temp = predictions.iloc[:,0].values.reshape(data_length,1)*CDFs[0].reshape(1,len(CDFs[0]))
            else:
                temp += predictions.iloc[:,e].values.reshape(data_length,1)*CDFs[e].reshape(1,len(CDFs[e]))
        else:
            temp += i*predictions.iloc[:,e].values.reshape(data_length,1) 
    return temp


# function wrapper for tree based xgboost GBM, with the softprob objective function. 
def train_tree_xgb(data,eta, gamma, max_d, min_child, subsamp, col_samp,num_classes, num_threads, num_over=3,eval_func=None):
    xg_train = xgb.DMatrix(data[0].values,label=data[1].values.ravel(),missing=np.nan)
    xg_val = xgb.DMatrix(data[2].values,label=data[3].values.ravel(),missing=np.nan)
    param = {}
    param['eta'] = eta
    param['gamma']  = gamma 
    param['max_depth'] = max_d
    param['min_child_weight'] = min_child
    param['subsample'] = subsamp
    param['colsample'] = col_samp
    param['nthread'] = num_threads
    param['num_class'] = num_classes
    param['objective'] = 'multi:softprob'
    param['eval_metric'] = 'mlogloss'
    param['silent'] = 1

    watchlist = [(xg_train, 'train'),(xg_val, 'test')]
    num_round = 10000
    bst = xgb.train(param, xg_train, num_round, watchlist, feval=eval_func,early_stopping_rounds=num_over)
    return bst

#this is faster with many features, but didn't end up improving score so its not used in final solution.
def train_linear_xgb(data,lmbda,alpha, lmbda_bias, num_classes,num_threads, num_over=3,eval_func=None):
    xg_train = xgb.DMatrix(data[0].values,label=data[1].values.ravel(),missing=np.nan)
    xg_val = xgb.DMatrix(data[2].values,label=data[3].values.ravel(),missing=np.nan)
    param1 = {}
    param1['objective'] = 'multi:softprob'
    param1['lambda'] = lmbda
    param1['alpha'] = alpha
    param1['lambda_bias'] = lmbda_bias
    param1['silent'] = 1
    param1['nthread'] = num_threads
    param1['num_class'] = num_classes
    param1['eval_metric'] = 'mlogloss'

    watchlist = [(xg_train, 'train'),(xg_val, 'test')]
    num_round =10000
    bst1 = xgb.train(param1,xg_train, num_round, watchlist, feval=eval_func, early_stopping_rounds=num_over)
    return bst1

# make predictions with a xgboost model on some data. 
def predict_bst(bst,validation):
    xg_val = xgb.DMatrix(validation.values,missing=np.nan)
    pred= bst.predict(xg_val);
    pred = pd.DataFrame(pred,index=validation.index)
    return pred


#Creates a xgboost object and loads model
def load_xgb_model(path_to_model):
    bst = xgb.Booster()
    bst.load_model(path_to_model)
    return bst

# load training data strictly greater then lower bound and strictly less the upperbound
# returns an cutoff value to be used in validation
def load_train_data(path_to_processed_data, lower_bound,upper_bound, offset_amount):
    full_train = pd.read_csv(path_to_processed_data + 'full_train.csv',index_col=0)
    hydro_train = pd.read_csv(path_to_processed_data + 'train_HydrometeorType_counts.csv',index_col=0)
    full_train = pd.concat([full_train, hydro_train],axis=1)

    train_counts = pd.read_csv(path_to_processed_data + 'train_counts.csv',index_col=0,header=None)
    train_counts.columns = ['cnt']

    actual_labels = pd.read_csv(path_to_processed_data + 'actual_labels.csv',index_col=0,header=None)
    actual_labels.columns = ['label']

    discrete_labels = discretize_labels(actual_labels)
    discrete_labels.index = actual_labels.index
    discrete_labels.columns = ['d_lab']

    
    #now take just the parts needed for this problem
    if lower_bound == upper_bound:
        train_counts = train_counts.query('cnt==@lower_bound')
    else:
        train_counts =  train_counts.query('cnt > @lower_bound and cnt < @upper_bound')

    discrete_labels = discrete_labels.reindex(train_counts.index)
    # a hack because indexes got mixed up somewhere. 


    offset = int(offset_amount*len(discrete_labels))

    #get rid of all the labels with rain amount >=70
    discrete_labels = discrete_labels.query('d_lab != 70')
    train_counts= train_counts.reindex(discrete_labels.index)
    full_train = full_train.reindex(discrete_labels.index)
    actual_labels = actual_labels.reindex(discrete_labels.index)

    train = full_train.reindex(train_counts.index) 
    actual_labels = actual_labels.reindex(train_counts.index)

    to_drop = ['DistanceToRadar_' + x for x in ['sum','mad','sem','krt','skw','max','min','std','mean','med','num_non_null']] + [ 'HybridScan_num_00','HybridScan_num_03','RadarQualityIndex_num_99']
    train = train.drop(to_drop, axis=1)
    return train, discrete_labels,actual_labels, offset

# loads the test data between lower and upper bound, 
def load_test_data(path_to_processed_data, lower_bound,upper_bound):
    full_test = pd.read_csv(path_to_processed_data + 'full_test.csv',index_col=0)
    hydro_test = pd.read_csv(path_to_processed_data + 'test_HydrometeorType_counts.csv',index_col=0)
    full_test = pd.concat([full_test, hydro_test],axis=1)

    test_counts = pd.read_csv(path_to_processed_data + 'test_counts.csv',index_col=0,header=None)
    test_counts.columns = ['cnt']

    train_counts = pd.read_csv(path_to_processed_data + 'train_counts.csv',index_col=0,header=None)
    train_counts.columns = ['cnt']

    actual_labels = pd.read_csv(path_to_processed_data + 'actual_labels.csv',index_col=0,header=None)
    actual_labels.columns = ['label']

    discrete_labels = discretize_labels(actual_labels)
    discrete_labels.index = actual_labels.index
    discrete_labels.columns = ['d_lab']

    
    #now take just the parts needed for this problem
    if lower_bound == upper_bound:
        test_counts =  test_counts.query('cnt==@lower_bound')
        train_counts = train_counts.query('cnt==@lower_bound')
    else:
        test_counts =  test_counts.query('cnt > @lower_bound and cnt < @upper_bound')
        train_counts =  train_counts.query('cnt > @lower_bound and cnt < @upper_bound')

    discrete_labels = discrete_labels.reindex(train_counts.index)
    #get rid of all the labels with rain amount >=70
    discrete_labels_all = discrete_labels.copy()
    discrete_labels = discrete_labels.query('d_lab != 70')
    train_counts= train_counts.reindex(discrete_labels.index)
    actual_labels = actual_labels.reindex(discrete_labels.index)

    test = full_test.reindex(test_counts.index) 
    actual_labels = actual_labels.reindex(train_counts.index)

    to_drop = ['DistanceToRadar_' + x for x in ['sum','mad','sem','krt','skw','max','min','std','mean','med','num_non_null']] + [ 'HybridScan_num_00','HybridScan_num_03','RadarQualityIndex_num_99']
    test = test.drop(to_drop, axis=1)
    return test, discrete_labels,actual_labels, discrete_labels_all

# creates solution for dataset 5 with more then 18 radar scans 
def make_prediction5(bst1,path,use_xtra_features=True):
    test, integer_labels, actual_labels, integer_labels_full = load_test_data(path,17,1000)

    if use_xtra_features:
        types = ['TimeToEnd','Reflectivity','Zdr','RR2','ReflectivityQC','RadarQualityIndex','RR3','RR1','Composite','RhoHV','HybridScan','LogWaterVolume']
        xtra_test = pd.DataFrame()
        for i in range(len(types)):
            xtra_test_temp = pd.read_csv(path+'test_'+types[i]+'18_199.csv',index_col=0)
            xtra_test = pd.concat([xtra_test,xtra_test_temp],axis=1)

        xtra_test = xtra_test.reindex(test.index)
        test= pd.concat([test, xtra_test],axis=1)

    pred_test = predict_bst(bst1,test)

    reduced_labels = aggregate_labels([[range(8,10),8],[range(10,14),9],[range(14,19),10],[range(19,70),11]], integer_labels).iloc[:,0] 
    cdfs_tst = make_cdf_list(8, 12, reduced_labels, integer_labels.iloc[:,0], 0)

    temp = create_full_predictions(cdfs_tst, pred_test)
    predictions = pd.DataFrame(temp, index=test.index)
    return predictions

def make_prediction4(bst1,path,use_xtra_features=True):
    test, integer_labels, actual_labels, integer_labels_full = load_test_data(path,7,18)

    if use_xtra_features:
        types = ['TimeToEnd','Reflectivity','Zdr','RR2','ReflectivityQC','RadarQualityIndex','RR3','RR1','Composite','RhoHV','HybridScan','LogWaterVolume']
        xtra_test = pd.DataFrame()
        for i in range(len(types)):
            xtra_test_temp = pd.read_csv(path+'test_'+types[i]+'8_17.csv',index_col=0)
            xtra_test = pd.concat([xtra_test,xtra_test_temp],axis=1)

        xtra_test = xtra_test.reindex(test.index)
        test= pd.concat([test, xtra_test],axis=1)

    pred_test = predict_bst(bst1,test)

    reduced_labels = aggregate_labels([[range(4,6),4],[range(6,70),5]], integer_labels).iloc[:,0]  
    cdfs_tst = make_cdf_list(4, 6, reduced_labels, integer_labels.iloc[:,0], 0)

    temp = create_full_predictions(cdfs_tst, pred_test)
    predictions = pd.DataFrame(temp, index=test.index)
    return predictions

def make_prediction3(bst1,path):
    test, integer_labels, actual_labels, integer_labels_full = load_test_data(path,3,8)

    pred_test = predict_bst(bst1,test)

    reduced_labels = aggregate_labels([[range(3,7),3],[range(7,70),4]], integer_labels).iloc[:,0]
    cdfs_tst = make_cdf_list(3, 5, reduced_labels, integer_labels_full.iloc[:,0], 0)

    temp = create_full_predictions(cdfs_tst, pred_test)
    predictions = pd.DataFrame(temp, index=test.index)
    return predictions


def make_prediction2(bst1,path):
    test, integer_labels, actual_labels, integer_labels_full = load_test_data(path,1,4)

    pred_test = predict_bst(bst1,test)


    reduced_labels = aggregate_labels([[range(3,7),3],[range(7,70),4]], integer_labels).iloc[:,0] 
    cdfs_tst = make_cdf_list(3, 5, reduced_labels, integer_labels.iloc[:,0], 0)
    temp = create_full_predictions(cdfs_tst, pred_test)
    predictions = pd.DataFrame(temp, index=test.index)
    return predictions


def make_prediction1(bst1,path):
    test, integer_labels, actual_labels, integer_labels_full = load_test_data(path,1,1)
    test = test.loc[:,test.mean() != -99999]

    pred_test = predict_bst(bst1,test)

    reduced_labels = aggregate_labels([[range(2,70),2]], integer_labels).iloc[:,0] 
    cdfs_tst = make_cdf_list(2, 3, reduced_labels, integer_labels.iloc[:,0], 0)

    temp = create_full_predictions(cdfs_tst, pred_test)
    predictions = pd.DataFrame(temp, index=test.index)
    return predictions

