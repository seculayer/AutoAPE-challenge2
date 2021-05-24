import pandas as pd
import numpy as np
import time

#counts the occurence of each value in hydrometoer columns
#returns a 13 column dataframe with counts for codes with more then one occurence
def make_hydrometeor_features(hydrometeor_data):
    hydro_feats = np.zeros([len(hydrometeor_data.index),15])
    for e,i in enumerate(hydrometeor_data.index):
        hydro = hydrometeor_data.iloc[e,:].value_counts()
        for j in hydro.index:
            hydro_feats[e,j] = hydro.loc[j]

    hydro_feats = pd.DataFrame(hydro_feats, index=hydrometeor_data.index)
    hydro_feats.columns = ['Hydro_'+str(x) for x in range(15)]

    #code 12 and 14 don't appear in train or test. 
    hydro_feats.drop(['Hydro_12','Hydro_14'],axis=1,inplace=True)
    return hydro_feats


# counts the times the error codes occur in each sample.
# after replaceing error codes with np.nan calculates various descriptive statistics
# fills nan's with single error code -99999
def make_stats_features(path_to_data,column_name,dataset_name):
    new_features = pd.DataFrame()
    data = pd.read_csv(path_to_data+dataset_name+'_'+column_name+'.csv',index_col=0)

    if (data == -99900.0).sum(axis=1).sum() > 100 :  
        new_features.loc[:,column_name + '_num_00'] = (data == -99900.0).sum(axis=1)    
    if (data == -99901.0).sum(axis=1).sum() > 100:  
        new_features.loc[:,column_name + '_num_01'] = (data == -99901.0).sum(axis=1)    
    if (data == -99903.0).sum(axis=1).sum() > 100:  
        new_features.loc[:,column_name + '_num_03'] = (data == -99903.0).sum(axis=1)    

    if column_name == 'RadarQualityIndex':
        new_features.loc[:,column_name + '_num_99'] = (data == 999.0).sum(axis=1)    
    data = data.replace([-99900.0,-99901.0,-99903.0,999.0],np.nan)

    new_features.loc[:,column_name+'_num_non_null'] = len(data.columns)-data.isnull().sum(axis=1)

    new_features.loc[:,column_name+'_mean'] = data.mean(axis=1, skipna=True)  
    new_features.loc[:,column_name+'_mean']  = new_features.loc[:,column_name+'_mean'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_std'] = data.std(axis=1)  
    new_features.loc[:,column_name+'_std']  = new_features.loc[:,column_name+'_std'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_min'] = data.min(axis=1)  
    new_features.loc[:,column_name+'_min']  = new_features.loc[:,column_name+'_min'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_max'] = data.max(axis=1)  
    new_features.loc[:,column_name+'_max']  = new_features.loc[:,column_name+'_max'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_med'] = data.median(axis=1)  
    new_features.loc[:,column_name+'_med']  = new_features.loc[:,column_name+'_med'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_skw'] = data.skew(axis=1)  
    new_features.loc[:,column_name+'_skw']  = new_features.loc[:,column_name+'_skw'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_krt'] = data.kurtosis(axis=1)  
    new_features.loc[:,column_name+'_krt']  = new_features.loc[:,column_name+'_krt'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_sem'] = data.sem(axis=1)  
    new_features.loc[:,column_name+'_sem']  = new_features.loc[:,column_name+'_sem'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_mad'] = data.mad(axis=1)  
    new_features.loc[:,column_name+'_mad']  = new_features.loc[:,column_name+'_mad'].replace(np.nan,-99999) 

    new_features.loc[:,column_name+'_sum'] = data.sum(axis=1)  
    new_features.loc[:,column_name+'_sum']  = new_features.loc[:,column_name+'_sum'].replace(np.nan,-99999) 
    
    new_features.index = data.index
    return new_features 

# process the raw train and test set, creates labels, the hydrometeor features, and then basic stats features.
def main():
    ## load the raw training data
    train = pd.read_csv('../input/train_2013.csv',index_col=0)
    actual_labels = train.Expected
    actual_labels.to_csv('../processed/actual_labels.csv')

    #remove kdp since it is all zeros
    for i in list(set(train.columns.tolist()) - set(['Kdp','Expected'])):
        temp_train = train[i].str.split(' ').apply(np.double).apply(pd.Series)
        temp_train.to_csv('../processed/train_'+i+'.csv')
        print 'train', i
    del train

    test = pd.read_csv('../input/test_2014.csv',index_col=0)
    for i in list(set(test.columns.tolist()) - set(['Kdp'])):
        temp_test = test[i].str.split(' ').apply(np.double).apply(pd.Series)
        temp_test.to_csv('../processed/test_'+ i + '.csv')
        print 'test', i
    del test

    trn_tte = pd.read_csv('../processed/train_TimeToEnd.csv',index_col=0) 
    tst_tte = pd.read_csv('../processed/test_TimeToEnd.csv',index_col=0) 

    trn_counts = len(trn_tte.columns)-trn_tte.isnull().sum(axis=1) 
    tst_counts = len(tst_tte.columns)-tst_tte.isnull().sum(axis=1) 

    tst_counts.to_csv('../processed/test_counts.csv')
    trn_counts.to_csv('../processed/train_counts.csv')

    train_hmt = pd.read_csv('../processed/train_HydrometeorType.csv',index_col=0)
    train_hydrometeor = make_hydrometeor_features(train_hmt)
    train_hydrometeor.to_csv('../processed/train_HydrometeorType_counts.csv')

    test_hmt = pd.read_csv('../processed/test_HydrometeorType.csv',index_col=0)
    test_hydrometeor = make_hydrometeor_features(test_hmt)
    test_hydrometeor.to_csv('../processed/test_HydrometeorType_counts.csv')


    test = pd.read_csv('../input/test_2014.csv',index_col=0)
    columns =  list(set(test.columns.tolist()) - set(['Kdp', 'HydrometeorType']))
    del test

    for i in ['train', 'test']:
        dataset = pd.DataFrame()
        for j in columns:
            temp_data = make_stats_features('../processed/',j,i)
            dataset = pd.concat([dataset, temp_data],axis=1)
            print i, j, 'stats'
        dataset.to_csv('../processed/full_'+i+'.csv')


if __name__ == "__main__":
    main()


