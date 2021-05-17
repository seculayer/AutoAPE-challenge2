import pandas as pd
import numpy as np
import random
import functions as fn 

def main():
    random.seed(11)
    np.random.seed(11)

    #set this to false for a faster training time
    use_xtra_features = True

    train_path = '../processed/'
    model_path = '../models/'

    model_name_suffix = '_final_subm'
    offset_amount = 0.07# fraction of the train set to use as hold out
    num_over = 2
    num_threads  = 7

    #generate trainset, labels, and test set based on the number of valid radar readings in the original dataset. 
    train, integer_labels,actual_labels, cutoff = fn.load_train_data(train_path,17,1000,offset_amount)

    reduced_labels = fn.aggregate_labels([[range(8,10),8],[range(10,14),9],[range(14,19),10],[range(19,70),11]], integer_labels).iloc[:,0] 

    if use_xtra_features:
        types = ['TimeToEnd','Reflectivity','Zdr','RR2','ReflectivityQC','RadarQualityIndex','RR3','RR1','Composite','RhoHV','HybridScan','LogWaterVolume']
        xtra_train = pd.DataFrame()
        for i in range(len(types)):
            xtra_train_temp = pd.read_csv(train_path+'train_'+types[i]+'18_199.csv', index_col=0)
            xtra_train = pd.concat([xtra_train,xtra_train_temp],axis=1)

        xtra_train = xtra_train.reindex(train.index)
        train= pd.concat([train, xtra_train],axis=1)

    data = (train.iloc[cutoff:,:],reduced_labels.iloc[cutoff:],train.iloc[:cutoff,:],reduced_labels.iloc[:cutoff])

    bst1 = fn.train_tree_xgb(data, 0.025, 2.5, 14, 85, .65, .5,12, num_threads, num_over)
    bst1.save_model(model_path+'bst5_1'+model_name_suffix)
    
if __name__ == "__main__":
    main()
